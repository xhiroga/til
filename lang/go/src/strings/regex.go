// 正規表現のメモ
package main

import (
  "fmt"
  "strings"
)


func main() {
	fmt.Println(strings.Replace("oink oink oink", "k", "ky", 2)) // oinky oinky oink
	fmt.Println(strings.Replace("oink oink oink", "oink", "moo", -1)) // moo moo moo
  fmt.Println(strings.Replace("/fmt", "/", "", 1)) //fmt ...つまりスラッシュなどもエスケープせずに使える。
  fmt.Println(strings.Replace("/net/http", "/", "", 1)) // net/http
}
