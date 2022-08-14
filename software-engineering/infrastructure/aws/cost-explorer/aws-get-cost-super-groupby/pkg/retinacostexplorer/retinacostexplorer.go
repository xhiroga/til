package retinacostexplorer

import (
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/costexplorer"
)

type CostExplorer interface {
	GetCostAndUsage(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error)
}

type RetinaCostExplorer struct {
	CostExplorer
}

func New(ce CostExplorer) *RetinaCostExplorer {
	return &RetinaCostExplorer{
		CostExplorer: ce,
	}
}

func (svc *RetinaCostExplorer) GetRetinaCostAndUsage(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error) {
	// group byの先頭2つを取り出す
	// 下処理段階ではコストの分類が欲しいだけなので、GranularityはMonthlyとみなす
	// Getを行う
	// TODO: ResultsByTimeをループし、通期に渡ってCostのAmountが0の項目を除外する
	// 次のGroupByの先頭2つを取り出す
	// 前回のResultsByTimeをループしながら再度再度を行う。ただし、前回のGroupByをFilterに追加し、かつ
	//

	// Outputの生成
	// もしGroup ByにLINKED_ACCOUNTが含まれていたら、OutputにDimensionValueAttributesを追加する
	// 引数のGroupDefinitionsをそのまま追加する

	return svc.GetCostAndUsageRecursively(input)
}

func (svc *RetinaCostExplorer) GetWholeCostAndUsage(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error) {
	output, err := svc.GetCostAndUsageRecursively(input)
	if err != nil {
		return nil, err
	}
	groupsArray := [][]*costexplorer.Group{}
	for _, resultsByTime := range output.ResultsByTime {
		groupsArray = append(groupsArray, resultsByTime.Groups)
	}
	return &costexplorer.GetCostAndUsageOutput{
		DimensionValueAttributes: output.DimensionValueAttributes,
		GroupDefinitions:         output.GroupDefinitions,
		NextPageToken:            output.NextPageToken,
		ResultsByTime: []*costexplorer.ResultByTime{
			{
				Estimated: output.ResultsByTime[0].Estimated,
				Groups:    totalMetricsByKey(groupsArray),
				TimePeriod: &costexplorer.DateInterval{
					End:   output.ResultsByTime[len(output.ResultsByTime)-1].TimePeriod.End,
					Start: output.ResultsByTime[0].TimePeriod.Start,
				},
				Total: output.ResultsByTime[0].Total,
			},
		},
	}, nil
}

type metrics map[string]*costexplorer.MetricValue
type groupedByKeys map[string]metrics

func totalMetricsByKey(groupsArray [][]*costexplorer.Group) []*costexplorer.Group {
	groupedByKeys := groupedByKeys{}

	for _, groups := range groupsArray {
		for _, group := range groups {
			key := strings.Join(StringsValues(group.Keys), ",")
			if _, ok := groupedByKeys[key]; !ok {
				groupedByKeys[key] = group.Metrics
			} else {
				for metricKey, metricValue := range group.Metrics {
					set, _ := strconv.ParseFloat(*groupedByKeys[key][metricKey].Amount, 64)
					new, _ := strconv.ParseFloat(*metricValue.Amount, 64)
					updatedAmount := strconv.FormatFloat(set+new, 'f', -1, 64)
					updatedMetricValue := &costexplorer.MetricValue{
						Amount: aws.String(updatedAmount),
						Unit:   groupedByKeys[key][metricKey].Unit,
					}
					groupedByKeys[key][metricKey] = updatedMetricValue
				}
			}
		}
	}
	return convertMetricsMapToGroups(groupedByKeys)
}

func convertMetricsMapToGroups(groupedByKeys groupedByKeys) []*costexplorer.Group {
	groups := []*costexplorer.Group{}
	for key, metrics := range groupedByKeys {
		keys := strings.Split(key, ",")

		group := &costexplorer.Group{
			Keys:    Strings(keys),
			Metrics: metrics,
		}
		groups = append(groups, group)
	}
	return groups
}

// GetCostAndUsage but get all pages.
func (svc *RetinaCostExplorer) GetCostAndUsageRecursively(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error) {
	output, err := svc.GetCostAndUsage(input)
	if err != nil {
		return nil, err
	}
	if output.NextPageToken != nil {
		input.NextPageToken = output.NextPageToken
		nextOutput, nextErr := svc.GetCostAndUsageRecursively(input)
		if nextErr != nil {
			return nil, nextErr
		}
		output.ResultsByTime = append(output.ResultsByTime, nextOutput.ResultsByTime...)
	}
	output.NextPageToken = nil
	return output, nil
}
