# Elixir

Erlang VM 上で動作する、Ruby 風の Syntax を持つ言語。

## 特徴

私が興味を持った特徴。

- アクターモデル（Scala における Akka)
- 並行処理
- Erlang のライブラリをオーバーヘッドなしに呼び出し可能
- Let It Crash
- OTP(OpenTelecomPlatform)フレームワークの利用
- パターンマッチ

### 並行処理とクラッシュについて

頑張ってエラー回避のコードを書くんじゃなくて、もうプロセスをクラッシュさせてしまえ、という考え方だという。  
一番例外が多そうなのは IO 処理だから、IO 処理とユースケースのロジックをプロセスレベルで分離してしまおう、ということだろう。確かに Java とかだとコーディング規則でエラー処理を入れるとかして対応する気がする。
https://qiita.com/HirofumiTamori/items/a3ee8eaeca76b43ae614

## Reference

https://www.slideshare.net/ohr486/elixirver2
