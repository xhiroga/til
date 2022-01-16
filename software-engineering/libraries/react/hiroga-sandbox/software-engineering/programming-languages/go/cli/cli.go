package main

import (
	"os"
	"fmt"
	"github.com/urfave/cli"
)

/*
	サマリ:
	urfave/cilは, os.Argsをいい感じに解釈して指定したアクションを実行してくれる構造体？を提供してくれる。
	アクションの指定方法は２通りあり、Action構造体に唯一のアクションを登録してくか、
	またはCommandsスライスにNameやActionなどをプロパティとして含むJsonを登録する。

	Helpに自動で表示するなどの都合を考えるとSliceで登録するのは妥当な感じがするが、個人的にはFlagの値によって処理がどのように分岐しているかがIF文と同じような見た目をしていて欲しいので
	使う人が多いツールを作るまでは使わない、かも。
*/
func main() {
	app := cli.NewApp()
	app.Name = "where"
	app.Usage = "show aws region options"

	app.Flags = []cli.Flag{
		cli.BoolFlag{
			Name: "list, l",
			Usage: "show available regions from 'aws ec2 describe-regions'",
		},
	}

	app.Action = func(c *cli.Context) error {
		fmt.Println("boom!")

		fmt.Printf("c.GlobalFlagNames()	: %+v\n", c.GlobalFlagNames())
		fmt.Printf("c.String(\"list\")	: %+v\n", c.String("list"))
		return nil
	}

	app.Run(os.Args)
}
