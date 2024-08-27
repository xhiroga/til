# Scala
JVMで動くオブジェクト指向言語かつ関数型言語。

## sbt
Scalaのビルドツール。build.sbtで設定可能。  
```console
sbt:scala> compile
sbt:scala> run

# bashから直接コマンドを実行することもできる
sbt package # target/scala-*.* ディレクトリに出力される
sbt assembly # 実行可能なjarファイルを作成。出力先は同上
```

# 参考
[sbt]
(https://www.scala-sbt.org/1.x/docs/index.html)