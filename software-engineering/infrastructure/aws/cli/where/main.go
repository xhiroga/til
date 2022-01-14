package main

import (
	"fmt"
	"flag"
	"sort"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
)

func main(){
	flag.Parse()
	cmd := flag.Arg(0)

	// 現在のRegionが知りたかったが、 aws configure listで可能だった...

	var regions []string
	if cmd == "list"{
		// ec2構造体を作る段階ではConfigのRegionプロパティは不要らしいが、DescribeRegionsする段階で無いとエラーになる（気がする）
		svc := ec2.New(session.New(), &aws.Config{Region: aws.String("us-east-1")})

		resp, err := svc.DescribeRegions(nil)
		if err != nil {
			panic(err)
		}
	
		for _, region := range resp.Regions {
			regions = append(regions, *region.RegionName)
		}
		sort.Strings(regions)

		for _, region := range regions {
			fmt.Println(region)
		}
		
		return 
	}


}