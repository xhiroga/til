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
			Granularity: aws.String("MONTHLY"),
			GroupBy: []*costexplorer.GroupDefinition{
				{
					Type: aws.String("DIMENSION"),
					Key:  aws.String("LINKED_ACCOUNT"),
				}, {
					Type: aws.String("DIMENSION"),
					Key:  aws.String("SERVICE"),
				},
			},
			Metrics: []*string{aws.String("BlendedCost")},
			TimePeriod: &costexplorer.DateInterval{
				Start: aws.String("2022-01-01"),
				End:   aws.String("2022-01-31"),
			},
		},
	)
	fmt.Println(res)
	fmt.Errorf("%v", err)
}
