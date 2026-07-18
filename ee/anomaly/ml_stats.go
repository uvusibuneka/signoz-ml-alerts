package anomaly

import (
	"math"
	"slices"
	"sort"
)

func percentileLinearInterpolation(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	if len(values) == 1 {
		return values[0]
	}

	sortedValues := slices.Clone(values)
	sort.Float64s(sortedValues)

	position := percentile * float64(len(sortedValues)-1)
	leftIndex := int(math.Floor(position))
	rightIndex := int(math.Ceil(position))
	if leftIndex == rightIndex {
		return sortedValues[leftIndex]
	}

	fraction := position - float64(leftIndex)
	return sortedValues[leftIndex] +
		fraction*(sortedValues[rightIndex]-sortedValues[leftIndex])
}

func aggregateMLAbsoluteScores(
	absoluteScores []float64,
	aggregation mlEnsembleAggregation,
) float64 {
	switch aggregation {
	case mlAggregationP75:
		return percentileNearestRank(absoluteScores, 0.75)
	case mlAggregationTop2Mean:
		return topNMean(absoluteScores, 2)
	default:
		return median(absoluteScores)
	}
}

func aggregateMLSignedScores(
	signedScores []float64,
	aggregation mlEnsembleAggregation,
) float64 {
	switch aggregation {
	case mlAggregationP75:
		return percentileNearestRank(signedScores, 0.75)
	case mlAggregationTop2Mean:
		return topNMeanByMagnitude(signedScores, 2)
	default:
		return median(signedScores)
	}
}

func percentileNearestRank(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sortedValues := slices.Clone(values)
	sort.Float64s(sortedValues)

	rank := int(math.Ceil(percentile * float64(len(sortedValues))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sortedValues) {
		rank = len(sortedValues)
	}

	return sortedValues[rank-1]
}

func topNMean(values []float64, count int) float64 {
	if len(values) == 0 {
		return 0
	}

	sortedValues := slices.Clone(values)
	sort.Float64s(sortedValues)
	if count > len(sortedValues) {
		count = len(sortedValues)
	}

	total := 0.0
	for index := len(sortedValues) - count; index < len(sortedValues); index++ {
		total += sortedValues[index]
	}

	return total / float64(count)
}

func topNMeanByMagnitude(values []float64, count int) float64 {
	if len(values) == 0 {
		return 0
	}

	sortedValues := slices.Clone(values)
	sort.Slice(sortedValues, func(left, right int) bool {
		return math.Abs(sortedValues[left]) < math.Abs(sortedValues[right])
	})

	if count > len(sortedValues) {
		count = len(sortedValues)
	}

	total := 0.0
	for index := len(sortedValues) - count; index < len(sortedValues); index++ {
		total += sortedValues[index]
	}

	return total / float64(count)
}

func standardDeviation(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	mean := 0.0
	for _, value := range values {
		mean += value
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, value := range values {
		difference := value - mean
		variance += difference * difference
	}
	variance /= float64(len(values))

	return math.Sqrt(variance)
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sortedValues := slices.Clone(values)
	sort.Float64s(sortedValues)

	middle := len(sortedValues) / 2
	if len(sortedValues)%2 == 1 {
		return sortedValues[middle]
	}

	return (sortedValues[middle-1] + sortedValues[middle]) / 2
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
