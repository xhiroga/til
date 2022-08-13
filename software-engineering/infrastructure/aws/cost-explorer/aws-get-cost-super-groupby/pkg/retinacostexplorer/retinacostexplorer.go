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

// GetCostAndUsage but can take "WHOLE" as a granularity.
// "WHOLE" is also a default value
func (svc *RetinaCostExplorer) GetWholeCostAndUsage(input *costexplorer.GetCostAndUsageInput) (*costexplorer.GetCostAndUsageOutput, error) {
	return svc.GetCostAndUsageRecursively(input)
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
