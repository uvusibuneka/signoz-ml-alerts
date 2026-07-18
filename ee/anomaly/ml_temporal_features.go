package anomaly

import (
	"math"
	"slices"
	"time"
)

func (provider *MLProvider) scoreTemporalPointLocked(
	seriesKey string,
	timestamp int64,
	value float64,
	fallbackScore float64,
) (float64, string, int) {
	state := provider.series[seriesKey]
	if state == nil {
		state = &mlSeriesState{}
		provider.series[seriesKey] = state
	}

	if timestamp <= state.lastTimestamp {
		return fallbackScore, "temporal_kmeans", len(state.temporalModels)
	}

	state.lastTimestamp = timestamp
	appendTemporalRawValue(state, timestamp, value, provider.config)
	feature, featureReady := appendTemporalFeature(state, timestamp, value, provider.config)

	state.lastFeatureSize = 0
	state.lastConsensusRatio = 0
	state.lastQuorumRatio = 0
	state.lastAnomalousFraction = 0
	state.lastUnanimous = false
	state.lastWeightedConsensus = false
	if featureReady {
		state.lastFeatureSize = len(feature.Vector)
	}

	provider.maybeTrainTemporalModel(state, timestamp)

	if !featureReady || len(state.temporalModels) < provider.config.TemporalMinimumModelsForConsensus {
		return fallbackScore, "temporal_kmeans", len(state.temporalModels)
	}

	consensus := evaluateTemporalConsensus(
		state.temporalModels,
		feature.Vector,
		timestamp,
		provider.config,
	)

	state.lastConsensusRatio = consensus.ConsensusRatio
	state.lastQuorumRatio = consensus.QuorumRatio
	state.lastAnomalousFraction = consensus.AnomalousFraction
	state.lastUnanimous = consensus.Unanimous
	state.lastWeightedConsensus = consensus.Weighted

	if !consensus.FinalAnomalous || consensus.QuorumRatio <= 1 {
		return 0, "temporal_kmeans", len(state.temporalModels)
	}

	direction := 1.0
	if isFinite(fallbackScore) && fallbackScore != 0 {
		direction = math.Copysign(1, fallbackScore)
	} else if len(state.temporalRawHistory) >= 2 {
		previousValue := state.temporalRawHistory[len(state.temporalRawHistory)-2].Value
		if value < previousValue {
			direction = -1
		}
	}

	score := direction * math.Min(4.0*consensus.QuorumRatio, provider.config.MaximumScore)
	return score, "temporal_kmeans", len(state.temporalModels)
}

func appendTemporalRawValue(
	state *mlSeriesState,
	timestamp int64,
	value float64,
	config mlConfig,
) {
	state.temporalRawHistory = append(state.temporalRawHistory, mlTimedValue{
		Timestamp: timestamp,
		Value:     value,
	})

	cutoffTimestamp := timestamp - int64(
		(config.TemporalTrainingWindow +
			time.Duration(config.TemporalMaximumModels+2)*config.TemporalTrainEvery).Milliseconds(),
	)

	trimIndex := 0
	for trimIndex < len(state.temporalRawHistory) &&
		state.temporalRawHistory[trimIndex].Timestamp < cutoffTimestamp {
		trimIndex++
	}

	if trimIndex > 0 {
		state.temporalRawHistory = slices.Clone(state.temporalRawHistory[trimIndex:])
	}
}

func appendTemporalFeature(
	state *mlSeriesState,
	timestamp int64,
	value float64,
	config mlConfig,
) (temporalFeaturePoint, bool) {
	if state.expectedInterval == 0 && len(state.temporalRawHistory) >= 2 {
		state.expectedInterval =
			state.temporalRawHistory[len(state.temporalRawHistory)-1].Timestamp -
				state.temporalRawHistory[len(state.temporalRawHistory)-2].Timestamp
	}

	if state.featureContext.HasLastRawValue &&
		state.expectedInterval > 0 &&
		len(state.temporalRawHistory) >= 2 {
		gap := timestamp - state.temporalRawHistory[len(state.temporalRawHistory)-2].Timestamp
		if gap*2 > state.expectedInterval*3 {
			resetTemporalFeatureContext(state)
		}
	}

	if !state.featureContext.HasLastRawValue {
		state.featureContext.HasLastRawValue = true
		state.featureContext.LastRawValue = value
		state.featureContext.RecentRawValues = append(state.featureContext.RecentRawValues, value)
		return temporalFeaturePoint{}, false
	}

	state.featureContext.LastRawValue = value
	state.featureContext.RecentRawValues = append(state.featureContext.RecentRawValues, value)

	requiredRawValues := config.TemporalDiffN + 1
	if len(state.featureContext.RecentRawValues) > requiredRawValues {
		state.featureContext.RecentRawValues =
			state.featureContext.RecentRawValues[len(state.featureContext.RecentRawValues)-requiredRawValues:]
	}
	if len(state.featureContext.RecentRawValues) < requiredRawValues {
		return temporalFeaturePoint{}, false
	}

	diff := value - state.featureContext.RecentRawValues[len(state.featureContext.RecentRawValues)-1-config.TemporalDiffN]
	state.featureContext.RecentDiffs = append(state.featureContext.RecentDiffs, diff)
	if len(state.featureContext.RecentDiffs) > config.TemporalSmoothN {
		state.featureContext.RecentDiffs =
			state.featureContext.RecentDiffs[len(state.featureContext.RecentDiffs)-config.TemporalSmoothN:]
	}
	if len(state.featureContext.RecentDiffs) < config.TemporalSmoothN {
		return temporalFeaturePoint{}, false
	}

	smoothed := meanFloat64s(state.featureContext.RecentDiffs)
	state.featureContext.RecentSmooths = append(state.featureContext.RecentSmooths, smoothed)

	requiredSmooths := config.TemporalLagN + 1
	if len(state.featureContext.RecentSmooths) > requiredSmooths {
		state.featureContext.RecentSmooths =
			state.featureContext.RecentSmooths[len(state.featureContext.RecentSmooths)-requiredSmooths:]
	}
	if len(state.featureContext.RecentSmooths) < requiredSmooths {
		return temporalFeaturePoint{}, false
	}

	vector := make(mlFeatureVector, 0, requiredSmooths)
	for index := len(state.featureContext.RecentSmooths) - 1; index >= 0; index-- {
		smoothValue := state.featureContext.RecentSmooths[index]
		if !isFinite(smoothValue) {
			return temporalFeaturePoint{}, false
		}
		vector = append(vector, smoothValue)
	}

	feature := temporalFeaturePoint{
		Timestamp: timestamp,
		Vector:    vector,
		SegmentID: state.featureContext.SegmentID,
	}

	state.temporalFeatureHistory = append(state.temporalFeatureHistory, feature)

	cutoffTimestamp := timestamp - int64(
		(config.TemporalTrainingWindow +
			time.Duration(config.TemporalMaximumModels+2)*config.TemporalTrainEvery).Milliseconds(),
	)

	trimIndex := 0
	for trimIndex < len(state.temporalFeatureHistory) &&
		state.temporalFeatureHistory[trimIndex].Timestamp < cutoffTimestamp {
		trimIndex++
	}
	if trimIndex > 0 {
		state.temporalFeatureHistory = state.temporalFeatureHistory[trimIndex:]
	}

	return feature, true
}

func resetTemporalFeatureContext(state *mlSeriesState) {
	state.featureContext = temporalFeatureContext{
		SegmentID: state.featureContext.SegmentID + 1,
	}
}

func meanFloat64s(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	total := 0.0
	for _, value := range values {
		total += value
	}

	return total / float64(len(values))
}

func calculateTemporalDifferences(rawValues []float64, diffN int) []float64 {
	if diffN <= 0 || len(rawValues) <= diffN {
		return nil
	}

	differences := make([]float64, 0, len(rawValues)-diffN)
	for index := diffN; index < len(rawValues); index++ {
		differences = append(differences, rawValues[index]-rawValues[index-diffN])
	}

	return differences
}

func calculateTemporalSmoothing(differences []float64, smoothN int) []float64 {
	if smoothN <= 0 || len(differences) < smoothN {
		return nil
	}

	smoothed := make([]float64, 0, len(differences)-smoothN+1)
	for end := smoothN; end <= len(differences); end++ {
		smoothed = append(smoothed, meanFloat64s(differences[end-smoothN:end]))
	}

	return smoothed
}

func buildTemporalLagFeatures(smoothed []float64, lagN int) []mlFeatureVector {
	required := lagN + 1
	if lagN < 0 || len(smoothed) < required {
		return nil
	}

	features := make([]mlFeatureVector, 0, len(smoothed)-required+1)
	for end := required; end <= len(smoothed); end++ {
		vector := make(mlFeatureVector, 0, required)
		for index := end - 1; index >= end-required; index-- {
			vector = append(vector, smoothed[index])
		}
		features = append(features, vector)
	}

	return features
}

func (provider *MLProvider) maybeTrainTemporalModel(
	state *mlSeriesState,
	timestamp int64,
) {
	if len(state.temporalFeatureHistory) == 0 {
		return
	}
	if state.lastTrainingTimestamp > 0 &&
		timestamp-state.lastTrainingTimestamp < provider.config.TemporalTrainEvery.Milliseconds() {
		return
	}

	latestFeature := state.temporalFeatureHistory[len(state.temporalFeatureHistory)-1]
	windowStartTimestamp := timestamp - provider.config.TemporalTrainingWindow.Milliseconds()

	trainingPoints := make([]temporalFeaturePoint, 0, len(state.temporalFeatureHistory))
	for _, feature := range state.temporalFeatureHistory {
		if feature.SegmentID != latestFeature.SegmentID {
			continue
		}
		if feature.Timestamp < windowStartTimestamp || feature.Timestamp > timestamp {
			continue
		}

		trainingPoints = append(trainingPoints, feature)
	}

	if len(trainingPoints) < 2 {
		return
	}
	if trainingPoints[0].Timestamp > windowStartTimestamp {
		return
	}

	model, ok := fitTemporalKMeansModel(trainingPoints, provider.config)
	if !ok {
		return
	}

	state.temporalModels = append(state.temporalModels, model)
	if len(state.temporalModels) > provider.config.TemporalMaximumModels {
		state.temporalModels =
			state.temporalModels[len(state.temporalModels)-provider.config.TemporalMaximumModels:]
	}

	state.lastTrainingTimestamp = timestamp
}
