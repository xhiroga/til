package ce

import (
	"github.com/aws/aws-sdk-go/service/costexplorer"
)

type CostExplorer interface {
	GetCostAndUsage(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error)
}

type RetinaCostExplorer struct {
	CostExplorer
}

func NewRetinaCostExplorer(ce CostExplorer) *RetinaCostExplorer {
	return &RetinaCostExplorer{
		CostExplorer: ce,
	}
}

func (svc *RetinaCostExplorer) GetRetinaCostAndUsage(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error) {
	return svc.GetCostAndUsage(input)
}

// func TestGetRetinaCostAndUsage(t *testing.T) {
// 	tests := []struct {
// 		name    string
// 		args    *costexplorer.GetCostAndUsageInput
// 		want    *costexplorer.GetCostAndUsageOutput
// 		wantErr bool
// 	}{}
// }
