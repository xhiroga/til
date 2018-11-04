package main

import (
	"flag"
	"fmt"
)

/*
	気になっていること
	* 引数がない、ということの判断ができるIFを提供しているか。
	→ フラグの数、非フラグの数をそれぞれNArg(), NFlagで提供している。

	* オプションを引数として扱うまではどのようなI/Fになっているか。
	ex. urfave/cliの場合はすでに構造体があり、ある意味コールバックとして処理のための関数を登録する形だった。
	→ flag構造体のプロパティを参照するポインタで受け取る。なお、func main内部で明示的にflag.Parse()を動かしてあげる必要はある。

	* その他注意点:
	args := flag.Args() このこ全部引数持ってくのでオプションと共存できない？
*/

var l = flag.Bool("l", false, "show available regions")

// flag.Parse() // ここは関数宣言の外なので宣言以外のことはできない。変数に代入する形で関数を使用することもできない（構造体からnew的なことはできるみたい）

func main() {
	flag.Parse()
	fmt.Println(*l)
}


// https://golang.org/pkg/flag/