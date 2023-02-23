# Go

## Go CLI

プログラムの実行方法は2種類ある

```shell
# 1. スクリプト言語のように実行
go run hello.go

# 2. コンパイルして実行可能ファイルを作成する
go build # go build hello.go のように、ファイル名を指定してもOK
./hello
```

ソースのフォーマットができる
```shell
go fmt # パッと見、ワード間のスペースの調整とかスペースのタブ置換とかできる
```

* パッケージ管理ができる
```shell
go install
go get
```

* 単体テストができる
```shell
go test
```

## References
- <https://www.udemy.com/go-the-complete-developers-guide/learn/v4/overview>
