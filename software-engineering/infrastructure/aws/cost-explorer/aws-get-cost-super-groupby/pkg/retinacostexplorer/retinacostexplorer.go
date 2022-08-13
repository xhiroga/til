package retinacostexplorer

import (
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
	// TODO
	return svc.GetCostAndUsage(input)
}
