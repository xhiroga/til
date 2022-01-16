package main

import "fmt"

/*
	実験: ポインタのポインタは作れる？
	予想: 作れると思う。 **で定義する？
*/

func main() {
	var i = 100
	var j *int
	var k **int
	var l ***int

	j = &i
	// k = &&i // これはエラーになるので...
	k = &j
	l = &k
	***l = ***l + 200

	fmt.Printf("%v\n", i)
	fmt.Printf("%v\n", j)
	fmt.Printf("%v\n", k)
	fmt.Printf("%v\n", l)

	// 答え: 自身に対するポインタはおそらく一つしかない（&を重ねるとエラーになる）


	/*
		実験: ポインタの値を参照するときはいくら*を重ねてもOK?
		予想: 大丈夫なんじゃない? 
		結果: ダメでした。*int と **int は明確に別物。
	*/

	// var m = 1000

	/*
	// これはエラーになる。
	var n *int
	n = &m
	fmt.Printf("%v\n", *******n)
	*/

	/*
	// これもエラーになる。
	var n *****int
	n = &m
	fmt.Printf("%v\n", *****n)
	*/
}