package retinacostexplorer

import (
	"reflect"
	"testing"

	mock_retinacostexplorer "xhiroga/retinacostexplorer/mock"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/costexplorer"
	"github.com/golang/mock/gomock"
)

func TestGetRetinaCostAndUsage(t *testing.T) {

	tests := []struct {
		name          string
		prepareMockFn func(mockCostExplorer *mock_retinacostexplorer.MockCostExplorer)
		args          *costexplorer.GetCostAndUsageInput
		want          *costexplorer.GetCostAndUsageOutput
		wantErr       bool
	}{
		{
			name: "normal (2 groups)",
			prepareMockFn: func(mockCostExplorer *mock_retinacostexplorer.MockCostExplorer) {
				input1 := &costexplorer.GetCostAndUsageInput{
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
				}
				output1 := &costexplorer.GetCostAndUsageOutput{
					DimensionValueAttributes: []*costexplorer.DimensionValuesWithAttributes{
						{
							Attributes: map[string]*string{
								"description": aws.String("dev"),
							},
							Value: aws.String("123456789012"),
						}, {
							Attributes: map[string]*string{
								"description": aws.String("prod"),
							},
							Value: aws.String("987654321098"),
						},
					},
					GroupDefinitions: []*costexplorer.GroupDefinition{{
						Type: aws.String("DIMENSION"),
						Key:  aws.String("LINKED_ACCOUNT"),
					}, {
						Type: aws.String("DIMENSION"),
						Key:  aws.String("USAGE_TYPE"),
					}},
					ResultsByTime: []*costexplorer.ResultByTime{{
						Estimated: aws.Bool(false),
						Groups: []*costexplorer.Group{
							{
								Keys: []*string{aws.String("123456789012"), aws.String("EC2 - Other")},
								Metrics: map[string]*costexplorer.MetricValue{
									"BlendedCost": {
										Amount: aws.String("3.716129016"),
										Unit:   aws.String("USD"),
									},
								},
							}, {
								Keys: []*string{aws.String("123456789012"), aws.String("AWS CloudTrail")},
								Metrics: map[string]*costexplorer.MetricValue{
									"BlendedCost": {
										Amount: aws.String("0"),
										Unit:   aws.String("USD"),
									},
								},
							}, {
								Keys: []*string{aws.String("987654321098"), aws.String("AWS CloudTrail")},
								Metrics: map[string]*costexplorer.MetricValue{
									"BlendedCost": {
										Amount: aws.String("2.90684"),
										Unit:   aws.String("USD"),
									},
								},
							},
						},
						TimePeriod: &costexplorer.DateInterval{
							End:   aws.String("2022-01-31"),
							Start: aws.String("2022-01-01"),
						},
						Total: map[string]*costexplorer.MetricValue{},
					}},
				}
				mockCostExplorer.EXPECT().GetCostAndUsage(input1).Return(output1, nil)
			},
			args: &costexplorer.GetCostAndUsageInput{
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
			want: &costexplorer.GetCostAndUsageOutput{
				DimensionValueAttributes: []*costexplorer.DimensionValuesWithAttributes{
					{
						Attributes: map[string]*string{
							"description": aws.String("dev"),
						},
						Value: aws.String("123456789012"),
					}, {
						Attributes: map[string]*string{
							"description": aws.String("prod"),
						},
						Value: aws.String("987654321098"),
					},
				},
				GroupDefinitions: []*costexplorer.GroupDefinition{{
					Type: aws.String("DIMENSION"),
					Key:  aws.String("LINKED_ACCOUNT"),
				}, {
					Type: aws.String("DIMENSION"),
					Key:  aws.String("USAGE_TYPE"),
				}},
				ResultsByTime: []*costexplorer.ResultByTime{{
					Estimated: aws.Bool(false),
					Groups: []*costexplorer.Group{
						{
							Keys: []*string{aws.String("123456789012"), aws.String("EC2 - Other")},
							Metrics: map[string]*costexplorer.MetricValue{
								"BlendedCost": {
									Amount: aws.String("3.716129016"),
									Unit:   aws.String("USD"),
								},
							},
						}, {
							Keys: []*string{aws.String("123456789012"), aws.String("AWS CloudTrail")},
							Metrics: map[string]*costexplorer.MetricValue{
								"BlendedCost": {
									Amount: aws.String("0"),
									Unit:   aws.String("USD"),
								},
							},
						}, {
							Keys: []*string{aws.String("987654321098"), aws.String("AWS CloudTrail")},
							Metrics: map[string]*costexplorer.MetricValue{
								"BlendedCost": {
									Amount: aws.String("2.90684"),
									Unit:   aws.String("USD"),
								},
							},
						},
					},
					TimePeriod: &costexplorer.DateInterval{
						End:   aws.String("2022-01-31"),
						Start: aws.String("2022-01-01"),
					},
					Total: map[string]*costexplorer.MetricValue{},
				}},
			},
			wantErr: false,
		},
		{
			name: "3 groups",
			prepareMockFn: func(mockCostExplorer *mock_retinacostexplorer.MockCostExplorer) {
				input1 := &costexplorer.GetCostAndUsageInput{
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
				}
				output1 := &costexplorer.GetCostAndUsageOutput{
					DimensionValueAttributes: []*costexplorer.DimensionValuesWithAttributes{
						{
							Attributes: map[string]*string{
								"description": aws.String("dev"),
							},
							Value: aws.String("123456789012"),
						}, {
							Attributes: map[string]*string{
								"description": aws.String("prod"),
							},
							Value: aws.String("987654321098"),
						},
					},
					GroupDefinitions: []*costexplorer.GroupDefinition{{
						Type: aws.String("DIMENSION"),
						Key:  aws.String("LINKED_ACCOUNT"),
					}, {
						Type: aws.String("DIMENSION"),
						Key:  aws.String("USAGE_TYPE"),
					}},
					ResultsByTime: []*costexplorer.ResultByTime{{
						Estimated: aws.Bool(false),
						Groups: []*costexplorer.Group{
							{
								Keys: []*string{aws.String("123456789012"), aws.String("EC2 - Other")},
								Metrics: map[string]*costexplorer.MetricValue{
									"BlendedCost": {
										Amount: aws.String("3.716129016"),
										Unit:   aws.String("USD"),
									},
								},
							}, {
								Keys: []*string{aws.String("123456789012"), aws.String("AWS CloudTrail")},
								Metrics: map[string]*costexplorer.MetricValue{
									"BlendedCost": {
										Amount: aws.String("0"),
										Unit:   aws.String("USD"),
									},
								},
							}, {
								Keys: []*string{aws.String("987654321098"), aws.String("AWS CloudTrail")},
								Metrics: map[string]*costexplorer.MetricValue{
									"BlendedCost": {
										Amount: aws.String("2.90684"),
										Unit:   aws.String("USD"),
									},
								},
							},
						},
						TimePeriod: &costexplorer.DateInterval{
							End:   aws.String("2022-01-31"),
							Start: aws.String("2022-01-01"),
						},
						Total: map[string]*costexplorer.MetricValue{},
					}},
				}
				mockCostExplorer.EXPECT().GetCostAndUsage(input1).Return(output1, nil)
			},
			args: &costexplorer.GetCostAndUsageInput{
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
					End:   aws.String("2022-01-31"),
				},
			},
			want: &costexplorer.GetCostAndUsageOutput{
				DimensionValueAttributes: []*costexplorer.DimensionValuesWithAttributes{
					{
						Attributes: map[string]*string{
							"description": aws.String("dev"),
						},
						Value: aws.String("123456789012"),
					}, {
						Attributes: map[string]*string{
							"description": aws.String("prod"),
						},
						Value: aws.String("987654321098"),
					},
				},
				GroupDefinitions: []*costexplorer.GroupDefinition{{
					Type: aws.String("DIMENSION"),
					Key:  aws.String("LINKED_ACCOUNT"),
				}, {
					Type: aws.String("DIMENSION"),
					Key:  aws.String("SERVICE"),
				}, {
					Type: aws.String("DIMENSION"),
					Key:  aws.String("USAGE_TYPE"),
				}},
				ResultsByTime: []*costexplorer.ResultByTime{{
					Estimated: aws.Bool(false),
					Groups: []*costexplorer.Group{
						{
							Keys: []*string{aws.String("123456789012"), aws.String("EC2 - Other"), aws.String("APN1-EBS:VolumeUsage.gp2")},
							Metrics: map[string]*costexplorer.MetricValue{
								"BlendedCost": {
									Amount: aws.String("3.716129016"),
									Unit:   aws.String("USD"),
								},
							},
						}, {
							Keys: []*string{aws.String("987654321098"), aws.String("AWS CloudTrail"), aws.String("APN1-PaidEventsRecorded")},
							Metrics: map[string]*costexplorer.MetricValue{
								"BlendedCost": {
									Amount: aws.String("1.37946"),
									Unit:   aws.String("USD"),
								},
							},
						}, {
							Keys: []*string{aws.String("987654321098"), aws.String("AWS CloudTrail"), aws.String("USE1-PaidEventsRecorded")},
							Metrics: map[string]*costexplorer.MetricValue{
								"BlendedCost": {
									Amount: aws.String("1.52738"),
									Unit:   aws.String("USD"),
								},
							},
						},
					},
					TimePeriod: &costexplorer.DateInterval{
						End:   aws.String("2022-01-31"),
						Start: aws.String("2022-01-01"),
					},
					Total: map[string]*costexplorer.MetricValue{},
				}},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCtrl := gomock.NewController(t)
			defer mockCtrl.Finish()

			mockCostExplorer := mock_retinacostexplorer.NewMockCostExplorer(mockCtrl)
			tt.prepareMockFn(mockCostExplorer)

			svc := New(mockCostExplorer)
			got, err := svc.GetRetinaCostAndUsage(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetRetinaCostAndUsage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetRetinaCostAndUsage() = %v, want %v", got, tt.want)
			}
		})
	}
}
