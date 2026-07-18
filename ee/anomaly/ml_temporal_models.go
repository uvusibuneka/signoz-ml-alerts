package anomaly

import (
	"math"
	"slices"
	"sort"
	"time"
)

func fitTemporalKMeansModel(
	trainingPoints []temporalFeaturePoint,
	config mlConfig,
) (temporalKMeansModel, bool) {
	vectors := make([]mlFeatureVector, 0, len(trainingPoints))
	for _, point := range trainingPoints {
		if !isFiniteFeatureVector(point.Vector) {
			continue
		}
		vectors = append(vectors, point.Vector)
	}

	centers, ok := fitKMeansND(
		vectors,
		config.TemporalClusterCount,
		config.KMeansIterations,
		config.MinimumScale,
	)
	if !ok {
		return temporalKMeansModel{}, false
	}

	distances := make([]float64, 0, len(vectors))
	for _, vector := range vectors {
		distances = append(distances, nearestCenterDistance(vector, centers))
	}
	if len(distances) == 0 {
		return temporalKMeansModel{}, false
	}

	cutoff := percentileLinearInterpolation(distances, config.TemporalDistanceQuantile)
	if cutoff < config.MinimumScale {
		cutoff = config.MinimumScale
	}

	return temporalKMeansModel{
		centers:        centers,
		distanceCutoff: cutoff,
		trainedAt:      time.UnixMilli(trainingPoints[len(trainingPoints)-1].Timestamp),
		windowStart:    time.UnixMilli(trainingPoints[0].Timestamp),
		windowEnd:      time.UnixMilli(trainingPoints[len(trainingPoints)-1].Timestamp),
		segmentID:      trainingPoints[len(trainingPoints)-1].SegmentID,
	}, true
}

func fitKMeansND(
	vectors []mlFeatureVector,
	clusterCount int,
	iterations int,
	minimumScale float64,
) ([]mlFeatureVector, bool) {
	if clusterCount < 1 || len(vectors) < clusterCount {
		return nil, false
	}

	dimension := len(vectors[0])
	if dimension == 0 {
		return nil, false
	}

	sortedVectors := slices.Clone(vectors)
	sort.Slice(sortedVectors, func(left, right int) bool {
		return compareFeatureVectorsLexicographically(
			sortedVectors[left],
			sortedVectors[right],
		) < 0
	})

	centers := make([]mlFeatureVector, clusterCount)
	for clusterIndex := range centers {
		quantileIndex := (2*clusterIndex + 1) * len(sortedVectors) / (2 * clusterCount)
		if quantileIndex >= len(sortedVectors) {
			quantileIndex = len(sortedVectors) - 1
		}
		centers[clusterIndex] = slices.Clone(sortedVectors[quantileIndex])
	}

	for iteration := 0; iteration < iterations; iteration++ {
		sums := make([][]float64, clusterCount)
		counts := make([]int, clusterCount)
		for clusterIndex := range sums {
			sums[clusterIndex] = make([]float64, dimension)
		}

		for _, vector := range vectors {
			clusterIndex := nearestCenterIndex(vector, centers)
			for dimensionIndex, value := range vector {
				sums[clusterIndex][dimensionIndex] += value
			}
			counts[clusterIndex]++
		}

		maxMovement := 0.0
		for clusterIndex := range centers {
			if counts[clusterIndex] == 0 {
				continue
			}

			updated := make(mlFeatureVector, dimension)
			for dimensionIndex := range updated {
				updated[dimensionIndex] =
					sums[clusterIndex][dimensionIndex] / float64(counts[clusterIndex])
			}

			movement := euclideanDistance(updated, centers[clusterIndex])
			if movement > maxMovement {
				maxMovement = movement
			}
			centers[clusterIndex] = updated
		}

		if maxMovement < minimumScale {
			break
		}
	}

	return centers, true
}

func scoreTemporalConsensus(
	models []temporalKMeansModel,
	vector mlFeatureVector,
	config mlConfig,
) (float64, bool) {
	result := evaluateTemporalConsensus(models, vector, 0, config)
	if !isFinite(result.QuorumRatio) {
		return 0, false
	}

	return result.QuorumRatio, result.FinalAnomalous
}

func evaluateTemporalConsensus(
	models []temporalKMeansModel,
	vector mlFeatureVector,
	currentTimestamp int64,
	config mlConfig,
) temporalConsensusResult {
	if len(models) == 0 {
		return temporalConsensusResult{}
	}

	ratioWeights := make([]weightedFloat64Value, 0, len(models))
	consensusRatio := math.Inf(1)
	totalWeight := 0.0
	anomalousWeight := 0.0
	unanimous := true
	weighted := config.TemporalRecencyHalfLife > 0

	for _, model := range models {
		ratio := temporalModelRatio(model, vector, config.MinimumScale)
		if !isFinite(ratio) {
			continue
		}
		if ratio < consensusRatio {
			consensusRatio = ratio
		}

		weight := temporalModelWeight(model, currentTimestamp, config.TemporalRecencyHalfLife)
		if !isFinite(weight) || weight <= 0 {
			continue
		}

		totalWeight += weight
		if ratio > 1 {
			anomalousWeight += weight
		} else {
			unanimous = false
		}

		ratioWeights = append(ratioWeights, weightedFloat64Value{
			value:  ratio,
			weight: weight,
		})
	}

	if len(ratioWeights) == 0 || totalWeight <= 0 || !isFinite(consensusRatio) {
		return temporalConsensusResult{}
	}

	anomalousFraction := anomalousWeight / totalWeight
	quorumRatio := weightedUpperQuantileFloat64(
		ratioWeights,
		1-config.TemporalConsensusFraction,
	)
	finalAnomalous := isFinite(quorumRatio) && quorumRatio > 1

	return temporalConsensusResult{
		ConsensusRatio:    consensusRatio,
		QuorumRatio:       quorumRatio,
		AnomalousFraction: anomalousFraction,
		FinalAnomalous:    finalAnomalous,
		Unanimous:         unanimous,
		Weighted:          weighted,
	}
}

func temporalModelWeight(
	model temporalKMeansModel,
	currentTimestamp int64,
	recencyHalfLife time.Duration,
) float64 {
	if recencyHalfLife <= 0 || currentTimestamp <= 0 {
		return 1
	}

	age := time.UnixMilli(currentTimestamp).Sub(model.trainedAt)
	if age <= 0 {
		return 1
	}

	return math.Exp(-math.Ln2 * age.Seconds() / recencyHalfLife.Seconds())
}

func weightedUpperQuantileFloat64(
	values []weightedFloat64Value,
	quantile float64,
) float64 {
	if len(values) == 0 {
		return 0
	}

	if quantile < 0 {
		quantile = 0
	}
	if quantile > 1 {
		quantile = 1
	}

	sortedValues := slices.Clone(values)
	sort.Slice(sortedValues, func(left, right int) bool {
		switch {
		case sortedValues[left].value < sortedValues[right].value:
			return true
		case sortedValues[left].value > sortedValues[right].value:
			return false
		default:
			return sortedValues[left].weight < sortedValues[right].weight
		}
	})

	totalWeight := 0.0
	for _, value := range sortedValues {
		if value.weight > 0 && isFinite(value.weight) {
			totalWeight += value.weight
		}
	}
	if totalWeight <= 0 {
		return 0
	}

	target := quantile * totalWeight
	cumulativeWeight := 0.0
	for _, value := range sortedValues {
		if value.weight <= 0 || !isFinite(value.weight) {
			continue
		}
		cumulativeWeight += value.weight
		if cumulativeWeight > target {
			return value.value
		}
	}

	return sortedValues[len(sortedValues)-1].value
}

func temporalModelRatio(
	model temporalKMeansModel,
	vector mlFeatureVector,
	minimumScale float64,
) float64 {
	distance := nearestCenterDistance(vector, model.centers)
	cutoff := model.distanceCutoff
	if cutoff <= minimumScale {
		cutoff = minimumScale
	}

	return distance / cutoff
}

func nearestCenterDistance(vector mlFeatureVector, centers []mlFeatureVector) float64 {
	if len(centers) == 0 {
		return 0
	}

	return euclideanDistance(vector, centers[nearestCenterIndex(vector, centers)])
}

func nearestCenterIndex(vector mlFeatureVector, centers []mlFeatureVector) int {
	nearestIndex := 0
	nearestDistance := math.Inf(1)
	for centerIndex, center := range centers {
		distance := euclideanDistance(vector, center)
		if distance < nearestDistance {
			nearestDistance = distance
			nearestIndex = centerIndex
		}
	}

	return nearestIndex
}

func euclideanDistance(left []float64, right []float64) float64 {
	limit := min(len(left), len(right))
	sum := 0.0
	for index := 0; index < limit; index++ {
		difference := left[index] - right[index]
		sum += difference * difference
	}

	return math.Sqrt(sum)
}

func isFiniteFeatureVector(vector mlFeatureVector) bool {
	if len(vector) == 0 {
		return false
	}

	for _, value := range vector {
		if !isFinite(value) {
			return false
		}
	}

	return true
}

func compareFeatureVectorsLexicographically(left []float64, right []float64) int {
	limit := min(len(left), len(right))
	for index := 0; index < limit; index++ {
		switch {
		case left[index] < right[index]:
			return -1
		case left[index] > right[index]:
			return 1
		}
	}

	switch {
	case len(left) < len(right):
		return -1
	case len(left) > len(right):
		return 1
	default:
		return 0
	}
}
