# jekyll-test

## URLはどうなる？

- URLの拡張子`.md`は`.html`になる。[^html]
- `.md`にアクセスすると、rawデータが見られる。[^md]

[^html]: https://xhiroga.github.io/jekyll-test/README.html
[^md]: https://xhiroga.github.io/jekyll-test/README.md

## リンクはどうなる？

普通に`<a>`タグに置き換えられる。

- [TEST.md](./TEST.md)

## アンダースコアやドットで始まるファイルは、リンクされていても表示されない？

表示されない。`jekyll`はリンクの有無で忖度をしない。

- [_TEST.md](./_TEST.md) - 表示されない (404)
- [.TEST.md](./.TEST.md) - 表示されない (404)

## `exclude`で指定したフォルダ内のファイルはどうなる？また、再帰的に効果がある？

`exclude`はもちろん効果があるが、再帰的ではない。

- [excluded/TEST.md](./excluded/TEST.md) - 表示されない (404)
- [sub/excluded/TEST.md](./sub/excluded/TEST.md) - 表示される！

## _config.yml は表示される？

`_`で始まるため、表示されない。

## リンクにしていないURLは、自動でリンクになる？

ならない。

## `details`は使える？

使えるが、`details`の中で更にコードブロックを使うことはできない。

<details><summary>サンプルコード</summary>

(上に空行が必要)

```ruby
puts 'Hello, World'
```
</details>

## 脚注は使える？

使える。

脚注[^1]

[^1]: これが表示されていると嬉しい。


## デバッグ

```shell
make
open http://localhost:4000
```

## 参考

- [Markdown記法 チートシート - Qiita](https://qiita.com/Qiita/items/c686397e4a0f4f11683d)

