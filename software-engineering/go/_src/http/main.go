package main

import (
  "net/http"
  "fmt"
  "html"
  "log"
)

func main(){

  // handlerを生成。もしRequestを受け取らなくて良いなら、http#HanldeでもOK
  http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
  	fmt.Fprintf(w, "Hello, %q", html.EscapeString(r.URL.Path)) // 要するにURLエンコードしてくれる
    // File print Formatted の略。要するに出力先を指定できるprint
  })

  log.Fatal(http.ListenAndServe(":8080", nil))
}

// 参考
// https://golang.org/pkg/net/http/
