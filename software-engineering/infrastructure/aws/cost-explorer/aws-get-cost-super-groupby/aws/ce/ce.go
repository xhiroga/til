package ce

import (
	"github.com/aws/aws-sdk-go/service/costexplorer"
)

type RetinaCostExplorer struct {
	*costexplorer.CostExplorer
}

func NewRetinaCostExplorer(ce *costexplorer.CostExplorer) *RetinaCostExplorer {
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
