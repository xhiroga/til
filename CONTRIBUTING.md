# About this guide

主に将来の自分（[@xhiroga](https://github.con/xhiroga)）に向けた、このリポジトリのメンテナンスのためのルールです。

## アクセシビリティ

- GitHub Pagesで公開するサイトは日本語読者（主に自分）向けとして、日本語で記載します。
- ソースコードのコメントは、ニュアンスが直感的に伝えられない場合を除いて英語で記載します。

## フォルダ構成

### 分類

```tree
...
├── computer-science        類目(MainClass) - 1次区分
│   ├── big-data            綱目(Division) - 2次区分
│   │   ├── _src
│   │   │   ├── getting-started
...
│   │   ├── README.md
...
├── README.md
...
```

- 分類は [NDC Vocabulary Definition](https://www.jla.or.jp/Portals/0/data/iinkai/bunrui/2_NDC%20Vocabulary.pdf) を参考にしています。
- ソースコードは `{綱目}/_src` 以下のディレクトリに配置します。Jekyllのビルド対象外にするためです。

:::note info
要目(Section)は、2023-02-19の更新で任意になりました。  
GitHub Pagesでのホスティングのためにマークダウンファイル(`*.md`)とソースコード(`_src/**`)を分けた結果、ネストが深くなることを懸念したものです。
:::

#### 類目(MainClass)

図書館の分類法を参考にして下さい。

- [国立国会図書館オンライン](https://ndlonline.ndl.go.jp/#!/)
- [日本十進分類法](https://www.ndl.go.jp/jp/data/NDC10code202006.pdf)
- [国立国会図書館分類表](https://www.ndl.go.jp/jp/data/catstandards/classification_subject/ndlc.html)
- [Library of Congress >> Books/Printed Material](https://www.loc.gov/books/?all=true)
- [Library of Congress Classification Outline](https://www.loc.gov/catdir/cpso/lcco/)
- [アメリカ議会図書館分類表(英) - Wikipedia](https://en.wikipedia.org/wiki/Library_of_Congress_Classification)
- [CIP](https://nces.ed.gov/ipeds/cipcode/browse.aspx?y=55) - アメリカ教育統計センターによる、教育プログラム分類体系

#### 綱目(Division)

類目によって分類方法が異なります。

##### Computer science

- [Computer science - Wikipedia](https://en.wikipedia.org/wiki/Computer_science)

##### Software Engineering

- [Zenn - Topics](https://zenn.dev/topics)
- [Qiita - Tags](https://qiita.com/tags)

## 開発

```shell
make
open http://localhost:4000
```

## 参考

- [About this guide  |  Google developer documentation style guide  |  Google Developers](https://developers.google.com/style)
- [js-primer/CONTRIBUTING.md at master · asciidwango/js-primer · GitHub](https://github.com/asciidwango/js-primer/blob/master/CONTRIBUTING.md)
