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
	return nil, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ## 関数のIn, Process, Out
// In:      通常のGetCostAndUsageInput
// Process: GetCostAndUsageを1回実行し、次のGroupByがなければ擬似GroupByを追加してOutputを返すし、そうでなければGroupを抽出した後Filterに変換して実行、結果を時間でZipして返す
// Out:     通常のGetCostAndUsageOutput
// ## 再帰の流れ
// GetCostAndUsageを実行
// 次のGroupByが残っていれば、今回のGroupByをFilterに追加して再帰的に実行する
// (1-1)
// GetCostAndUsageをFilter付きで実行
// 次のGroupByが残っていれば、今回のGroupByを時間方向でSumした後、Filterに追加して再帰的にGroup回数分実行する
// (2-1)
// GetCostAndUsageをFilter付きで実行
// 次のGroupByが無ければ、FilterのValueを擬似GroupByとして追加してOutputを返す
// (2-2)
// GetCostAndUsageをFilter付きで実行
// 次のGroupByが無ければ、FilterのValueを擬似GroupByとして追加してOutputを返す
// ...
// 擬似GroupByが追加されたそれぞれ異なるDimensionのOutputの配列が取得されるので、時間でZipして返却
// (1-2)
// 省略
// 擬似GroupByが追加されたそれぞれ異なるDimensionのOutputの配列が取得されるので、時間でZipして返却
func (svc *RetinaCostExplorer) getCostAndUsageWithPseudoFilters(base *costexplorer.GetCostAndUsageInput, groupBy []*costexplorer.GroupDefinition, pseudoGroupFilter *costexplorer.Expression) (*costexplorer.GetCostAndUsageOutput, error) {
	current := groupBy[:min(2, len(groupBy))]
	next := groupBy[min(2, len(groupBy)):]

	filter := base.Filter
	if pseudoGroupFilter != nil {
		filter = &costexplorer.Expression{
			And: []*costexplorer.Expression{filter, base.Filter},
		}
	}

	output, err := svc.GetCostAndUsageRecursively(&costexplorer.GetCostAndUsageInput{
		// TODO
		Filter:  filter,
		GroupBy: current,
	})
	if err != nil {
		return nil, err
	}

	if len(next) > 0 {
		recursed := [][]*costexplorer.ResultByTime{}
		result := sumResultsThroughTime(output.ResultsByTime)
		for _, group := range result.Groups {
			nextFilter := extractFilterFromGroup(current, group)
			output, err := svc.getCostAndUsageWithPseudoFilters(base, next, nextFilter)
			if err != nil {
				return nil, err
			}
			recursed = append(recursed, output.ResultsByTime)
		}
		return &costexplorer.GetCostAndUsageOutput{
			// TODO
			ResultsByTime: zipResultsByTime(recursed),
		}, nil
	} else {
		return &costexplorer.GetCostAndUsageOutput{
			// TODO
			ResultsByTime: prependPseudoGroupsFromFilter(output.ResultsByTime, pseudoGroupFilter),
		}, nil
	}
}

func prependPseudoGroupsFromFilter(results []*costexplorer.ResultByTime, pseudoGroupFilter *costexplorer.Expression) []*costexplorer.ResultByTime {
	// TODO
	return []*costexplorer.ResultByTime{}
}

func zipResultsByTime(resultsArray [][]*costexplorer.ResultByTime) []*costexplorer.ResultByTime {
	// TODO
	return []*costexplorer.ResultByTime{}
}

// TODO: Groupの抽出、的な名前に変更したい
func sumResultsThroughTime(results []*costexplorer.ResultByTime) *costexplorer.ResultByTime {
	// TODO: resultsをまたいで同じ日時のGroupsが存在する場合、Uniqueチェックがないので多重計上になってしまう問題がある。
	groupsArray := [][]*costexplorer.Group{}
	for _, result := range results {
		groupsArray = append(groupsArray, result.Groups)
	}
	return &costexplorer.ResultByTime{
		Estimated: results[0].Estimated,
		Groups:    totalMetricsByKey(groupsArray),
		TimePeriod: &costexplorer.DateInterval{
			End:   results[len(results)-1].TimePeriod.End,
			Start: results[0].TimePeriod.Start,
		},
		Total: results[0].Total,
	}
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

func extractFiltersFromGroups(groupDefs []*costexplorer.GroupDefinition, groups []*costexplorer.Group) []*costexplorer.Expression {
	filters := []*costexplorer.Expression{}
	for _, group := range groups {
		filters = append(filters, extractFilterFromGroup(groupDefs, group))
	}
	return filters
}

func extractFilterFromGroup(groupDefs []*costexplorer.GroupDefinition, group *costexplorer.Group) *costexplorer.Expression {
	exps := []*costexplorer.Expression{}
	for i, groupDef := range groupDefs {
		exp := &costexplorer.Expression{
			Dimensions: &costexplorer.DimensionValues{
				Key: groupDef.Key,
				Values: []*string{
					group.Keys[i],
				},
				MatchOptions: []*string{
					aws.String("EQUALS"),
				},
			},
		}
		exps = append(exps, exp)
	}
	return &costexplorer.Expression{
		And: exps,
	}
}

// TODO: recursiveを他で使うので、 GetCostAndUsage but get all pages. っぽい名前に直す
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
