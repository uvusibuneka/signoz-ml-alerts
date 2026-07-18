package anomaly

import (
	"math"
	"slices"
	"sort"

	qbtypes "github.com/SigNoz/signoz/pkg/types/querybuildertypes/querybuildertypesv5"
)

func shouldLearnMLSample(
	value *qbtypes.TimeSeriesValue,
	score float64,
	config mlConfig,
) bool {
	if value == nil || value.Partial || !isFinite(value.Value) {
		return false
	}
	if !isFinite(score) {
		return true
	}

	return math.Abs(score) < config.LearningScoreLimit
}

func trimMLHistory(state *mlSeriesState, config mlConfig) {
	if len(state.values) <= config.MaximumSamples {
		return
	}

	start := len(state.values) - config.MaximumSamples
	state.values = slices.Clone(state.values[start:])
}

func buildKMeansEnsemble(values []float64, config mlConfig) []kMeansModel {
	if len(values) < config.CriticalMass {
		return nil
	}

	models := make([]kMeansModel, 0, len(config.TrainingWindows)*len(config.ClusterCounts))
	for _, window := range config.TrainingWindows {
		if len(values) < window {
			continue
		}

		trainingValues := values[len(values)-window:]
		for _, clusterCount := range config.ClusterCounts {
			model, ok := fitKMeans1D(trainingValues, clusterCount, config)
			if ok {
				models = append(models, model)
			}
		}
	}

	return models
}

func fitKMeans1D(
	values []float64,
	clusterCount int,
	config mlConfig,
) (kMeansModel, bool) {
	if clusterCount < 1 || len(values) < clusterCount {
		return kMeansModel{}, false
	}

	cleanValues := make([]float64, 0, len(values))
	for _, value := range values {
		if isFinite(value) {
			cleanValues = append(cleanValues, value)
		}
	}
	if len(cleanValues) < clusterCount {
		return kMeansModel{}, false
	}

	sortedValues := slices.Clone(cleanValues)
	sort.Float64s(sortedValues)

	centroids := make([]float64, clusterCount)
	for clusterIndex := range centroids {
		quantileIndex := (2*clusterIndex + 1) * len(sortedValues) / (2 * clusterCount)
		if quantileIndex >= len(sortedValues) {
			quantileIndex = len(sortedValues) - 1
		}
		centroids[clusterIndex] = sortedValues[quantileIndex]
	}

	for iteration := 0; iteration < config.KMeansIterations; iteration++ {
		sums := make([]float64, clusterCount)
		counts := make([]int, clusterCount)

		for _, value := range cleanValues {
			clusterIndex := nearestCentroid(value, centroids)
			sums[clusterIndex] += value
			counts[clusterIndex]++
		}

		maxMovement := 0.0
		for clusterIndex := range centroids {
			if counts[clusterIndex] == 0 {
				continue
			}

			updated := sums[clusterIndex] / float64(counts[clusterIndex])
			movement := math.Abs(updated - centroids[clusterIndex])
			if movement > maxMovement {
				maxMovement = movement
			}
			centroids[clusterIndex] = updated
		}

		if maxMovement < config.MinimumScale {
			break
		}
	}

	globalScale := standardDeviation(cleanValues)
	scaleFloor := math.Max(globalScale*config.ScaleFloorFactor, config.MinimumScale)

	squaredDistances := make([]float64, clusterCount)
	counts := make([]int, clusterCount)
	for _, value := range cleanValues {
		clusterIndex := nearestCentroid(value, centroids)
		difference := value - centroids[clusterIndex]
		squaredDistances[clusterIndex] += difference * difference
		counts[clusterIndex]++
	}

	scales := make([]float64, clusterCount)
	for clusterIndex := range scales {
		if counts[clusterIndex] < 2 {
			scales[clusterIndex] = scaleFloor
			continue
		}

		scale := math.Sqrt(squaredDistances[clusterIndex] / float64(counts[clusterIndex]))
		scales[clusterIndex] = math.Max(scale, scaleFloor)
	}

	return kMeansModel{centroids: centroids, scales: scales}, true
}

func scoreKMeansEnsemble(
	models []kMeansModel,
	value float64,
	fallbackScore float64,
	config mlConfig,
) float64 {
	if len(models) == 0 || !isFinite(value) {
		return fallbackScore
	}

	signedScores := make([]float64, 0, len(models))
	absoluteScores := make([]float64, 0, len(models))

	for _, model := range models {
		clusterIndex := nearestCentroid(value, model.centroids)
		scale := model.scales[clusterIndex]
		if scale < config.MinimumScale {
			scale = config.MinimumScale
		}

		score := (value - model.centroids[clusterIndex]) / scale
		if !isFinite(score) {
			continue
		}

		signedScores = append(signedScores, score)
		absoluteScores = append(absoluteScores, math.Abs(score))
	}

	if len(absoluteScores) == 0 {
		return fallbackScore
	}

	magnitude := aggregateMLAbsoluteScores(absoluteScores, config.Aggregation)
	direction := 1.0

	if isFinite(fallbackScore) && fallbackScore != 0 {
		direction = math.Copysign(1, fallbackScore)
	} else if aggregateMLSignedScores(signedScores, config.Aggregation) < 0 {
		direction = -1
	}

	return direction * math.Min(magnitude, config.MaximumScore)
}

func (provider *MLProvider) scoreRawPointLocked(
	seriesKey string,
	timestamp int64,
	value float64,
	fallbackScore float64,
) (float64, string, int) {
	state := provider.series[seriesKey]
	if state == nil {
		state = &mlSeriesState{
			values: make([]float64, 0, provider.config.CriticalMass),
		}
		provider.series[seriesKey] = state
	}

	state.lastConsensusRatio = 0
	state.lastQuorumRatio = 0
	state.lastAnomalousFraction = 0
	state.lastUnanimous = false
	state.lastWeightedConsensus = false
	state.lastFeatureSize = 0

	models := buildKMeansEnsemble(state.values, provider.config)
	finalScore := fallbackScore
	if len(models) > 0 && isFinite(value) {
		finalScore = scoreKMeansEnsemble(models, value, fallbackScore, provider.config)
	}

	if timestamp <= state.lastTimestamp {
		mode := "zscore_warmup"
		if len(models) > 0 {
			mode = "kmeans_ensemble"
		}
		return finalScore, mode, len(models)
	}

	state.lastTimestamp = timestamp
	metricValue := &qbtypes.TimeSeriesValue{
		Timestamp: timestamp,
		Value:     value,
	}

	if shouldLearnMLSample(metricValue, finalScore, provider.config) {
		state.values = append(state.values, value)
		trimMLHistory(state, provider.config)
	}

	mode := "zscore_warmup"
	if len(models) > 0 {
		mode = "kmeans_ensemble"
	}

	return finalScore, mode, len(models)
}

func nearestCentroid(value float64, centroids []float64) int {
	nearestIndex := 0
	nearestDistance := math.Inf(1)
	for centroidIndex, centroid := range centroids {
		distance := math.Abs(value - centroid)
		if distance < nearestDistance {
			nearestDistance = distance
			nearestIndex = centroidIndex
		}
	}

	return nearestIndex
}
