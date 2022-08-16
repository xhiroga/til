package retinacostexplorer

import (
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
	base := &costexplorer.GetCostAndUsageInput{
		Granularity: input.Granularity,
		Metrics:     input.Metrics,
		TimePeriod:  input.TimePeriod,
	}
	return svc.getCostAndUsageRecursively(base, input.GroupBy, nil)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (svc *RetinaCostExplorer) getCostAndUsageRecursively(base *costexplorer.GetCostAndUsageInput, groupBy []*costexplorer.GroupDefinition, pseudoGroupFilter *costexplorer.Expression) (*costexplorer.GetCostAndUsageOutput, error) {
	current := groupBy[:min(2, len(groupBy))]
	next := groupBy[min(2, len(groupBy)):]

	filter := base.Filter
	if pseudoGroupFilter != nil {
		filter = &costexplorer.Expression{
			And: []*costexplorer.Expression{filter, base.Filter},
		}
	}

	output, err := svc.GetCostAndUsageAllPages(&costexplorer.GetCostAndUsageInput{
		// TODO
		Filter:  filter,
		GroupBy: current,
	})
	if err != nil {
		return nil, err
	}

	if len(next) > 0 {
		recursed := [][]*costexplorer.ResultByTime{}
		groups := extractUniqueGroups(output.ResultsByTime)
		for _, group := range groups {
			nextFilter := extractFilterFromGroup(current, group)
			output, err := svc.getCostAndUsageRecursively(base, next, nextFilter)
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
	timeToGroups := map[string][]*costexplorer.Group{}
	for _, results := range resultsArray {
		for _, result := range results {
			key := *result.TimePeriod.Start + "/" + *result.TimePeriod.End
			timeToGroups[key] = append(timeToGroups[key], result.Groups...)
		}
	}
	results := []*costexplorer.ResultByTime{}
	for key, groups := range timeToGroups {
		time := strings.Split(key, "/")
		results = append(results, &costexplorer.ResultByTime{
			// TODO: EstimatedとTotalは考慮しない
			Groups: groups,
			TimePeriod: &costexplorer.DateInterval{
				Start: aws.String(time[0]),
				End:   aws.String(time[1]),
			},
		})
	}
	return results
}

func extractUniqueGroups(results []*costexplorer.ResultByTime) []*costexplorer.Group {
	groups := []*costexplorer.Group{}
	for _, result := range results {
		for _, group := range result.Groups {
			if !containsGroup(groups, group) {
				groups = append(groups, group)
			}
		}
	}
	return groups
}

// TODO: パフォーマンス改善の余地がありそう
func containsGroup(groups []*costexplorer.Group, group *costexplorer.Group) bool {
	for _, g := range groups {
		if strings.Join(StringsValues(g.Keys), ",") == strings.Join(StringsValues(group.Keys), ",") {
			return true
		}
	}
	return false
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

func (svc *RetinaCostExplorer) GetCostAndUsageAllPages(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error) {
	output, err := svc.GetCostAndUsage(input)
	if err != nil {
		return nil, err
	}
	if output.NextPageToken != nil {
		input.NextPageToken = output.NextPageToken
		nextOutput, nextErr := svc.GetCostAndUsageAllPages(input)
		if nextErr != nil {
			return nil, nextErr
		}
		output.ResultsByTime = append(output.ResultsByTime, nextOutput.ResultsByTime...)
	}
	output.NextPageToken = nil
	return output, nil
}
