package main

import (
	"fmt"
)

func main() {
	var en = "Hello" // OK
	var cn string = "nihao" // 文法的には通るが、 string = まで右辺として解釈されてしまう。
	jp := "こんにちは" // ここではOKだが、Goの内部では代入→代入先がなければ宣言、と見なされるらしく、関数宣言外では代入ができないことからこの文法は関数宣言外では使えない。

	fmt.Printf(cn, en, jp)
}
