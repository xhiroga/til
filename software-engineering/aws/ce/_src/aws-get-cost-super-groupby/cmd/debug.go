package main

import (
	"fmt"

	"xhiroga/retinacostexplorer/pkg/retinacostexplorer"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/costexplorer"
)

func main() {
	sess := session.Must(session.NewSessionWithOptions(session.Options{}))
	svc := retinacostexplorer.New(
		costexplorer.New(
			sess,
		),
	)
	res, err := svc.GetRetinaCostAndUsage(
		&costexplorer.GetCostAndUsageInput{
			Filter: &costexplorer.Expression{
				And: []*costexplorer.Expression{{
					Dimensions: &costexplorer.DimensionValues{
						Key: aws.String("REGION"),
						Values: []*string{
							aws.String("ap-northeast-1"),
						},
						MatchOptions: []*string{
							aws.String("EQUALS"),
						},
					},
				}},
			},
			Granularity: aws.String("MONTHLY"),
			GroupBy: []*costexplorer.GroupDefinition{
				{
					Type: aws.String("DIMENSION"),
					Key:  aws.String("LINKED_ACCOUNT"),
				}, {
					Type: aws.String("DIMENSION"),
					Key:  aws.String("SERVICE"),
				}, {
					Type: aws.String("DIMENSION"),
					Key:  aws.String("USAGE_TYPE"),
				},
			},
			Metrics: []*string{aws.String("BlendedCost")},
			TimePeriod: &costexplorer.DateInterval{
				Start: aws.String("2022-01-01"),
				End:   aws.String("2022-02-28"),
			},
		},
	)
	fmt.Println(res)
	fmt.Errorf("%v", err)
}
