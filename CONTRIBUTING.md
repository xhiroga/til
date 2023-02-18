# About this guide

主に将来の自分（[@xhiroga](https://github.con/xhiroga)）に向けた、このリポジトリのメンテナンスのためのルールです。

## アクセシビリティ

- GitHub Pagesで公開するサイトは日本語読者（主に自分）向けとして、日本語で記載します。
- ソースコードのコメントは、ニュアンスが直感的に伝えられない場合を除いて英語で記載します。

## フォルダ構成

### 分類

```tree
...
├── computer-science        MainClass / 類目（1次区分）
│   ├── big-data            Division / 綱目（2次区分）
│   │   ├── apache-hadoop   Section / 要目（3次区分）
...
├── README.md
...
```

The name of classification is from [NDC Vocabulary Definition](https://www.jla.or.jp/Portals/0/data/iinkai/bunrui/2_NDC%20Vocabulary.pdf).

:::note info
要目(Section)は、2023-02-18の更新で無くなりました。  
GitHub Pagesでのホスティングのためにマークダウンファイル(`*.md`)とソースコード(`_src/**`)を分けた結果、ネストが深くなることを懸念したものです。
:::


### MainClass

Often set up with reference to library classification.

- [国立国会図書館オンライン](https://ndlonline.ndl.go.jp/#!/)
- [日本十進分類法](https://www.ndl.go.jp/jp/data/NDC10code202006.pdf)
- [国立国会図書館分類表](https://www.ndl.go.jp/jp/data/catstandards/classification_subject/ndlc.html)
- [Library of Congress >> Books/Printed Material](https://www.loc.gov/books/?all=true)
- [Library of Congress Classification Outline](https://www.loc.gov/catdir/cpso/lcco/)
- [アメリカ議会図書館分類表(英) - Wikipedia](https://en.wikipedia.org/wiki/Library_of_Congress_Classification)
- [CIP](https://nces.ed.gov/ipeds/cipcode/browse.aspx?y=55) - アメリカ教育統計センターによる、教育プログラム分類体系

### Division

Reference to category of computer science.

- [Computer science \- Wikipedia](https://en.wikipedia.org/wiki/Computer_science)

### Section

Alphabetical order.





## 開発

```shell
make
open http://localhost:4000
```

## 脆弱性対応





## 参考

- [About this guide  |  Google developer documentation style guide  |  Google Developers](https://developers.google.com/style)
- [js-primer/CONTRIBUTING.md at master · asciidwango/js-primer · GitHub](https://github.com/asciidwango/js-primer/blob/master/CONTRIBUTING.md)
