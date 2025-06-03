# About this guide

このリポジトリに貢献するためのルールです。

## 文章の内容

- 文書がREADME.mdであるとき、その内容はフォルダ名に従います。
- 文書構造はMECEを心がけてください。
　- Bad

    ```aws/README.md
    # EC2のコスト削減
    ```

  - Good

    ```aws/README.md
    # AWS
    ## Amazon EC2
    ```

- 常識的な内容は積極的に省いてください。
　- Bad

    ```cURL/README.md
    # cURL
    cURLは、CLIからHTTPリクエストを送信するためのツールです。
    ## 使い方
    ```

  - Good

    ```cURL/README.md
    # cURL
    ## TIPS
    - JSONの送信方法
    ```

## アクセシビリティ

- GitHub Pagesで公開するサイトは日本語読者（主に自分）向けとして、日本語で記載します。
- ソースコードのコメントは、ニュアンスが直感的に伝えられない場合を除いて英語で記載します。

## フォーマット

[Markdownのスタイルガイド](./styleguides/markdown.md)を参照してください。

### 引用文献フォーマット

CSの分野の学会などでメジャーなAPA形式を採用します。[Google Scholar Button](https://chrome.google.com/webstore/detail/google-scholar-button/ldipcbpaocekfooobnbcddclnhejkcpn)を用います。

## フォルダ構成

- 文書（レポートとソースコード）は、その分類（類目・綱目）に応じて2階層のフォルダを作成し、そのフォルダ内に記載します。
- 文書の追加時は、次のとおりフォルダを判断してください。
  - キーワードで検索を行い、書籍の分類を参照してください。
    - 国立国会図書館サーチ(<https://ndlsearch.ndl.go.jp/search?cs=bib&keyword={検索ワード}>)

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
- [日本十進分類法](https://www.libnet.pref.okayama.jp/shiryou/ndc/index.htm)
  - [蔵書検索 | 東京都立図書館](https://catalog.library.metro.tokyo.lg.jp/winj/opac/search-detail.do)
  - [DDC から NDC への記号変換](https://contents.nii.ac.jp/sites/default/files/2020-03/WATARAI.pdf): 英訳あり
- [国立国会図書館分類表](https://www.ndl.go.jp/jp/data/catstandards/classification_subject/ndlc.html)
- [Library of Congress >> Books/Printed Material](https://www.loc.gov/books/?all=true)
- [Library of Congress Classification Outline](https://www.loc.gov/catdir/cpso/lcco/)
- [アメリカ議会図書館分類表(英) - Wikipedia](https://en.wikipedia.org/wiki/Library_of_Congress_Classification)
- [CIP](https://nces.ed.gov/ipeds/cipcode/browse.aspx?y=55) - アメリカ教育統計センターによる、教育プログラム分類体系

:::note info
日本十進分類法を採用している図書館の例を検討中です。東京都立中央図書館, 大阪市立/府立中央図書館, 滋賀県立図書館などのサイトを確認しましたが、URL直アクセスによる検索ができないか遅かったため、掲示していません。
:::

#### 綱目(Division)

類目によって分類方法が異なります。

##### Computer science

- [Computer science - Wikipedia](https://en.wikipedia.org/wiki/Computer_science)

##### Software Engineering

- [Zenn - Topics](https://zenn.dev/topics)
- [Qiita - Tags](https://qiita.com/tags)
- [AWS CLI Command Reference](https://docs.aws.amazon.com/cli/latest/reference/) - AWSのセクションのフォルダ名に用いる。

## 開発

```shell
make
open http://localhost:4000
```

## コーディング規則

- Pythonの依存性管理およびビルドには`uv`を用います。トラブルシューティングに`pip`を用いるのは避けてください。
- どうしてもという場合は`uv pip`でpip相当のコマンドが利用できます。
- 変数名などから機能が分かる場合は、コメントなどで2重に説明する必要はありません。
- Makefileのターゲット名から分かることは、READMEに書く必要がありません。
- むしろ変数名・定数名で分かるよう、表現力の高いコードを心がけてください。
- チャット上の対話が日本語であっても、プログラム中のコメントは同じファイルのそれ以外のコメントに倣ってください。
- コミットメッセージは英語で書いてください。

## 参考

- [About this guide  |  Google developer documentation style guide  |  Google Developers](https://developers.google.com/style)
- [js-primer/CONTRIBUTING.md at master · asciidwango/js-primer · GitHub](https://github.com/asciidwango/js-primer/blob/master/CONTRIBUTING.md)
