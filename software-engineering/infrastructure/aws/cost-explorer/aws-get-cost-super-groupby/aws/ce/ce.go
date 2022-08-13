package ce

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/costexplorer"
)

func Get() {
	sess := session.Must(session.NewSessionWithOptions(session.Options{}))
	svc := costexplorer.New(
		sess,
	)
	res, err := svc.GetCostAndUsage(
		&costexplorer.GetCostAndUsageInput{
			Granularity: aws.String("MONTHLY"),
			GroupBy:     []*costexplorer.GroupDefinition{{Key: aws.String("SERVICE"), Type: aws.String("DIMENSION")}},
			Metrics:     []*string{aws.String("BlendedCost")},
			// NextPageToken: new(string),
			TimePeriod: &costexplorer.DateInterval{Start: aws.String("2022-01-01"), End: aws.String("2022-08-31")},
		},
	)
	fmt.Println(res)
	fmt.Errorf("%v", err)
}
