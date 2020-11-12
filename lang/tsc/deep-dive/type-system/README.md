# Type System

型エイリアス、型アノテーション、型アサーション、型ガード、型推論...似てるのばっかりで参る！勉強しよう。

## How to run

```sh
deno run src/types.ts
```

## ここが謎

変数宣言と型エイリアスでトークンが重複しても構わないらしい。

```ts
let Direction
type Direction
```

よく考えるとたしかに問題ない。なぜなら型は型アノテーション(:の方)(型ガードもアノテーションのうち)と型アサーション(as の方。ASsertion だけに)以外で書かないから。（typeof 演算子を利用した場合の比較で出てくるかと思ったが、なんと typeof が返すのは文字列型なので問題なし。 === で比較しているから普通の型、と覚えればいいか。）

https://typescript-jp.gitbook.io/deep-dive/type-system
