package main // 実行可能パッケージを作成するための特別なパッケージ名。また、func main()が必須となる。
// その他のパッケージ（再利用可能パッケージ）の名前は自由。

import "fmt"

func main() {
	// フォーマットIOパッケージのPrintf関数を実行
	fmt.Printf("hello, world\n")
	fmt.Println("Hi, there!")
}
