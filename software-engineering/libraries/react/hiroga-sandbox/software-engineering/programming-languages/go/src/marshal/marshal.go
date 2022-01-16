package main

import (
    "encoding/json"
    "fmt"
)

type Greet struct {
    Id int
    Greet string
    Country string
}

const greet_jp = `{
    "id": 1,
    "greet": "こんにちは",
    "country": "Japan"
}`

// 文字列のjsonを型に変換するのがUnmarshal(非整列化)
func main() {
    jsonBytes := ([]byte)(greet_jp)
    parsedData := new(Greet)

    err := json.Unmarshal(jsonBytes, parsedData)
    if err != nil {
        fmt.Println("json Unmarshal error:", err)
        return
    }

    fmt.Println(parsedData.Greet)
}