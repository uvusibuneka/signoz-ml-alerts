package anomaly

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/SigNoz/signoz/pkg/querier"
	qbtypes "github.com/SigNoz/signoz/pkg/types/querybuildertypes/querybuildertypesv5"
	"github.com/SigNoz/signoz/pkg/valuer"
)

const (
	mlCriticalMass       = 256
	mlMaximumSamples     = 2048
	mlKMeansIterations   = 32
	mlLearningScoreLimit = 3.5
	mlMaximumScore       = 100.0
	mlMinimumScale       = 1e-9
)

type mlEnsembleAggregation string

const (
	mlAggregationMedian   mlEnsembleAggregation = "median"
	mlAggregationP75      mlEnsembleAggregation = "p75"
	mlAggregationTop2Mean mlEnsembleAggregation = "top2_mean"
)

type mlAlgorithmMode string

const (
	mlAlgorithmRawEnsemble mlAlgorithmMode = "raw_ensemble"
	mlAlgorithmTemporal    mlAlgorithmMode = "temporal"
)

type mlConfig struct {
	AlgorithmMode                     mlAlgorithmMode
	CriticalMass                      int
	MaximumSamples                    int
	KMeansIterations                  int
	LearningScoreLimit                float64
	MaximumScore                      float64
	MinimumScale                      float64
	TrainingWindows                   []int
	ClusterCounts                     []int
	ScaleFloorFactor                  float64
	Aggregation                       mlEnsembleAggregation
	TemporalDiffN                     int
	TemporalSmoothN                   int
	TemporalLagN                      int
	TemporalClusterCount              int
	TemporalTrainingWindow            time.Duration
	TemporalTrainEvery                time.Duration
	TemporalMaximumModels             int
	TemporalMinimumModelsForConsensus int
	TemporalDistanceQuantile          float64
	TemporalConsensusFraction         float64
	TemporalRecencyHalfLife           time.Duration
}

var defaultMLConfig = mlConfig{
	AlgorithmMode:      mlAlgorithmTemporal,
	CriticalMass:       mlCriticalMass,
	MaximumSamples:     mlMaximumSamples,
	KMeansIterations:   mlKMeansIterations,
	LearningScoreLimit: mlLearningScoreLimit,
	MaximumScore:       mlMaximumScore,
	MinimumScale:       mlMinimumScale,
	TrainingWindows: []int{
		256,
		512,
		1024,
	},
	ClusterCounts: []int{
		2,
		3,
		4,
	},
	ScaleFloorFactor:                  0.05,
	Aggregation:                       mlAggregationMedian,
	TemporalDiffN:                     1,
	TemporalSmoothN:                   3,
	TemporalLagN:                      5,
	TemporalClusterCount:              2,
	TemporalTrainingWindow:            6 * time.Hour,
	TemporalTrainEvery:                3 * time.Hour,
	TemporalMaximumModels:             18,
	TemporalMinimumModelsForConsensus: 18,
	TemporalDistanceQuantile:          0.99,
	TemporalConsensusFraction:         1.0,
	TemporalRecencyHalfLife:           0,
}

type MLProvider struct {
	baseProvider Provider
	querier      querier.Querier
	logger       *slog.Logger

	mu     sync.Mutex
	series map[string]*mlSeriesState
	config mlConfig
}

type mlSeriesState struct {
	values                 []float64
	lastTimestamp          int64
	temporalRawHistory     []mlTimedValue
	temporalFeatureHistory []temporalFeaturePoint
	temporalModels         []temporalKMeansModel
	lastTrainingTimestamp  int64
	expectedInterval       int64
	featureContext         temporalFeatureContext
	lastConsensusRatio     float64
	lastQuorumRatio        float64
	lastAnomalousFraction  float64
	lastUnanimous          bool
	lastWeightedConsensus  bool
	lastFeatureSize        int
}

type kMeansModel struct {
	centroids []float64
	scales    []float64
}

type mlTimedValue struct {
	Timestamp int64
	Value     float64
}

type mlFeatureVector []float64

type temporalFeaturePoint struct {
	Timestamp int64
	Vector    mlFeatureVector
	SegmentID int
}

type temporalFeatureContext struct {
	HasLastRawValue bool
	LastRawValue    float64
	RecentRawValues []float64
	RecentDiffs     []float64
	RecentSmooths   []float64
	SegmentID       int
}

type temporalKMeansModel struct {
	centers        []mlFeatureVector
	distanceCutoff float64
	trainedAt      time.Time
	windowStart    time.Time
	windowEnd      time.Time
	segmentID      int
}

type temporalConsensusResult struct {
	ConsensusRatio    float64
	QuorumRatio       float64
	AnomalousFraction float64
	FinalAnomalous    bool
	Unanimous         bool
	Weighted          bool
}

type weightedFloat64Value struct {
	value  float64
	weight float64
}

var _ Provider = (*MLProvider)(nil)

func NewMLProvider(
	baseProvider Provider,
	querier querier.Querier,
	logger *slog.Logger,
) *MLProvider {
	return newMLProviderWithConfig(
		baseProvider,
		querier,
		logger,
		defaultMLConfig,
	)
}

func newMLProviderWithConfig(
	baseProvider Provider,
	querier querier.Querier,
	logger *slog.Logger,
	config mlConfig,
) *MLProvider {
	if logger == nil {
		logger = slog.Default()
	}

	return &MLProvider{
		baseProvider: baseProvider,
		querier:      querier,
		logger:       logger,
		series:       make(map[string]*mlSeriesState),
		config:       normalizeMLConfig(config),
	}
}

func (provider *MLProvider) GetAnomalies(
	ctx context.Context,
	orgID valuer.UUID,
	request *AnomaliesRequest,
) (*AnomaliesResponse, error) {
	if request == nil {
		return nil, fmt.Errorf("anomalies request is required")
	}
	if request.Params == nil {
		return nil, fmt.Errorf("anomaly query parameters are required")
	}
	if provider.baseProvider == nil {
		return nil, fmt.Errorf("base anomaly provider is required")
	}

	requestCopy := *request
	response, err := provider.baseProvider.GetAnomalies(ctx, orgID, &requestCopy)
	if err != nil {
		return nil, fmt.Errorf("run base anomaly provider: %w", err)
	}

	provider.applyMLScores(ctx, response)
	return response, nil
}

func (provider *MLProvider) applyMLScores(
	ctx context.Context,
	response *AnomaliesResponse,
) {
	if response == nil {
		return
	}

	provider.mu.Lock()
	defer provider.mu.Unlock()

	for _, result := range response.Results {
		if result == nil {
			continue
		}

		for aggregationIndex, aggregation := range result.Aggregations {
			if aggregation == nil {
				continue
			}

			seriesCount := min(len(aggregation.Series), len(aggregation.AnomalyScores))
			for seriesIndex := 0; seriesIndex < seriesCount; seriesIndex++ {
				metricSeries := aggregation.Series[seriesIndex]
				scoreSeries := aggregation.AnomalyScores[seriesIndex]
				if metricSeries == nil || scoreSeries == nil {
					continue
				}

				key := mlSeriesKey(result.QueryName, aggregationIndex, seriesIndex, metricSeries)

				state := provider.series[key]
				if state == nil {
					state = &mlSeriesState{
						values: make([]float64, 0, provider.config.CriticalMass),
					}
					provider.series[key] = state
				}

				samplesBefore := len(state.values)
				mode := "zscore_warmup"
				modelCount := 0

				valueCount := min(len(metricSeries.Values), len(scoreSeries.Values))
				for valueIndex := 0; valueIndex < valueCount; valueIndex++ {
					metricValue := metricSeries.Values[valueIndex]
					scoreValue := scoreSeries.Values[valueIndex]
					if metricValue == nil || scoreValue == nil {
						continue
					}

					finalScore, sampleMode, sampleModels := provider.scoreSinglePointLocked(
						key,
						metricValue.Timestamp,
						metricValue.Value,
						scoreValue.Value,
					)
					scoreValue.Value = finalScore
					mode = sampleMode
					modelCount = sampleModels
				}

				provider.logger.InfoContext(
					ctx,
					"running ML anomaly provider",
					slog.String("ml.implementation", mode),
					slog.Int("ml.samples_before", samplesBefore),
					slog.Int("ml.samples_after", len(state.values)),
					slog.Int("ml.models", modelCount),
					slog.String("ml.series", key),
					slog.Int("ml.models_available", len(state.temporalModels)),
					slog.Int("ml.models_required", provider.config.TemporalMinimumModelsForConsensus),
					slog.Float64("ml.consensus_ratio", state.lastConsensusRatio),
					slog.Float64("ml.consensus_fraction", provider.config.TemporalConsensusFraction),
					slog.Float64("ml.anomalous_fraction", state.lastAnomalousFraction),
					slog.Float64("ml.quorum_ratio", state.lastQuorumRatio),
					slog.Bool("ml.unanimous", state.lastUnanimous),
					slog.Bool("ml.weighted_consensus", state.lastWeightedConsensus),
					slog.Int("ml.feature_size", state.lastFeatureSize),
					slog.Duration("ml.training_window", provider.config.TemporalTrainingWindow),
					slog.Duration("ml.train_every", provider.config.TemporalTrainEvery),
					slog.Duration("ml.recency_half_life", provider.config.TemporalRecencyHalfLife),
				)
			}
		}
	}
}

func mlSeriesKey(
	queryName string,
	aggregationIndex int,
	seriesIndex int,
	series *qbtypes.TimeSeries,
) string {
	labelsKey := qbtypes.GetUniqueSeriesKey(series.Labels)
	if labelsKey == "" {
		labelsKey = fmt.Sprintf("series_index=%d", seriesIndex)
	}

	return fmt.Sprintf("%s|%d|%s", queryName, aggregationIndex, labelsKey)
}

func normalizeMLConfig(config mlConfig) mlConfig {
	normalized := config

	switch normalized.AlgorithmMode {
	case mlAlgorithmRawEnsemble, mlAlgorithmTemporal:
	default:
		normalized.AlgorithmMode = defaultMLConfig.AlgorithmMode
	}

	if normalized.CriticalMass <= 0 {
		normalized.CriticalMass = defaultMLConfig.CriticalMass
	}
	if normalized.MaximumSamples <= 0 {
		normalized.MaximumSamples = defaultMLConfig.MaximumSamples
	}
	if normalized.KMeansIterations <= 0 {
		normalized.KMeansIterations = defaultMLConfig.KMeansIterations
	}
	if normalized.LearningScoreLimit <= 0 {
		normalized.LearningScoreLimit = defaultMLConfig.LearningScoreLimit
	}
	if normalized.MaximumScore <= 0 {
		normalized.MaximumScore = defaultMLConfig.MaximumScore
	}
	if normalized.MinimumScale <= 0 {
		normalized.MinimumScale = defaultMLConfig.MinimumScale
	}

	if len(normalized.TrainingWindows) == 0 {
		normalized.TrainingWindows = append([]int(nil), defaultMLConfig.TrainingWindows...)
	} else {
		normalized.TrainingWindows = append([]int(nil), normalized.TrainingWindows...)
	}

	if len(normalized.ClusterCounts) == 0 {
		normalized.ClusterCounts = append([]int(nil), defaultMLConfig.ClusterCounts...)
	} else {
		normalized.ClusterCounts = append([]int(nil), normalized.ClusterCounts...)
	}

	if normalized.ScaleFloorFactor <= 0 {
		normalized.ScaleFloorFactor = defaultMLConfig.ScaleFloorFactor
	}

	switch normalized.Aggregation {
	case mlAggregationMedian, mlAggregationP75, mlAggregationTop2Mean:
	default:
		normalized.Aggregation = defaultMLConfig.Aggregation
	}

	if normalized.TemporalDiffN <= 0 {
		normalized.TemporalDiffN = defaultMLConfig.TemporalDiffN
	}
	if normalized.TemporalSmoothN <= 0 {
		normalized.TemporalSmoothN = defaultMLConfig.TemporalSmoothN
	}
	if normalized.TemporalLagN <= 0 {
		normalized.TemporalLagN = defaultMLConfig.TemporalLagN
	}
	if normalized.TemporalClusterCount <= 0 {
		normalized.TemporalClusterCount = defaultMLConfig.TemporalClusterCount
	}
	if normalized.TemporalClusterCount > 16 {
		normalized.TemporalClusterCount = 16
	}
	if normalized.TemporalTrainingWindow <= 0 {
		normalized.TemporalTrainingWindow = defaultMLConfig.TemporalTrainingWindow
	}
	if normalized.TemporalTrainEvery <= 0 {
		normalized.TemporalTrainEvery = defaultMLConfig.TemporalTrainEvery
	}
	if normalized.TemporalMaximumModels <= 0 {
		normalized.TemporalMaximumModels = defaultMLConfig.TemporalMaximumModels
	}
	if normalized.TemporalMaximumModels > 64 {
		normalized.TemporalMaximumModels = 64
	}
	if normalized.TemporalMinimumModelsForConsensus <= 0 {
		normalized.TemporalMinimumModelsForConsensus = defaultMLConfig.TemporalMinimumModelsForConsensus
	}
	if normalized.TemporalMinimumModelsForConsensus > normalized.TemporalMaximumModels {
		normalized.TemporalMinimumModelsForConsensus = normalized.TemporalMaximumModels
	}
	if normalized.TemporalDistanceQuantile <= 0 || normalized.TemporalDistanceQuantile > 1 {
		normalized.TemporalDistanceQuantile = defaultMLConfig.TemporalDistanceQuantile
	}
	if normalized.TemporalConsensusFraction < 0.5 || normalized.TemporalConsensusFraction > 1.0 {
		normalized.TemporalConsensusFraction = defaultMLConfig.TemporalConsensusFraction
	}
	if normalized.TemporalRecencyHalfLife < 0 {
		normalized.TemporalRecencyHalfLife = 0
	}

	return normalized
}

func (provider *MLProvider) scoreSinglePointForSeries(
	seriesKey string,
	timestamp int64,
	value float64,
	fallbackScore float64,
) float64 {
	provider.mu.Lock()
	defer provider.mu.Unlock()

	finalScore, _, _ := provider.scoreSinglePointLocked(
		seriesKey,
		timestamp,
		value,
		fallbackScore,
	)

	return finalScore
}

func (provider *MLProvider) scoreSinglePointLocked(
	seriesKey string,
	timestamp int64,
	value float64,
	fallbackScore float64,
) (float64, string, int) {
	switch provider.config.AlgorithmMode {
	case mlAlgorithmTemporal:
		return provider.scoreTemporalPointLocked(seriesKey, timestamp, value, fallbackScore)
	default:
		return provider.scoreRawPointLocked(seriesKey, timestamp, value, fallbackScore)
	}
}
