# Markdown Style Guide

## Style Rules

### Footnote

脚注の識別子には、番号ではなく出典や著者情報を用いる。詳細は検討中。

### Heading (Style Rules)

- URLのパスに用いられることがあるので、簡潔な英語を推奨する。
- 英語の場合、大文字で始める。入力する手間より、見出しが単語ではなく文になったときの小文字始まりの違和感が大きいため。

### Multilingual

用語の日英併記が便利そうな場合、次のように書き表す。`日本語 (ENG, English)`のようにする。

- 簡潔データ構造 (succinct data structure)
- 画像の認識・理解シンポジウム (MIRU, Meeting on Image Recognition and Understanding)

表記の統一にあたっては、次の要素を考慮した。

- ダブルクリック時の選択範囲
  - 半角カッコを使うと、日本語・英語のどちらもクリックできる
  - 全角括弧を使うと、日本語・英語をつなげて1つの単語として認識する
- マークダウンのヘッダーからリンクに変換する際の規則
  - 半角スペースは半角ハイフンに変換される。
  - 全角スペースは無視される。
  - 括弧は全角・半角を問わず無視される。

## Formatting Rules

### Heading (Formatting Rules)

- (`##` と `--` のどちらを用いるか、調整中...)

### Code Block

ふつう、言語識別子 (language identifier)には、言語名または拡張子名を用いることができる。しかし、拡張子名は言語名に対するエイリアスとして設定されている場合が多い。したがって、言語識別子としては言語名を用いる。判断にあたっては、次のライブラリを参考にした。

| Name                        | Use Cases                   |
| --------------------------- | --------------------------- |
| highlight.js[^highlight.js] | VSCode                      |
| linguist[^linguist]         | GitHub[^github]             |
| rouge[^rouge]               | GitHub Pages[^github_pages] |
| VSCode[^vscode]             |                             |

[^github]: [Creating and highlighting code blocks | GitHub](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks#syntax-highlighting)
[^github_pages]: [About GitHub Pages and Jekyll | GitHub](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll#syntax-highlighting)
[^highlight.js]: [SUPPORTED_LANGUAGES.md | highlight.js](https://github.com/highlightjs/highlight.js/blob/main/SUPPORTED_LANGUAGES.md)
[^linguist]: [languages.yml | linguist](https://github.com/github-linguist/linguist/blob/master/lib/linguist/languages.yml)
[^rouge]: [Languages.md | rouge](https://github.com/rouge-ruby/rouge/wiki/List-of-supported-languages-and-lexers)
[^vscode]: [markdownEngine.ts | VSCode](https://github.com/microsoft/vscode/blob/main/extensions/markdown-language-features/src/markdownEngine.ts)

- Shell Scriptの言語識別子は`shell`とする。調査したすべての環境で利用でき、かつシェルの種類を特定しない書き方であるため。
- コンソールの出力に対する言語識別子は`console`とする。linguistとrougeで共通で利用できるため。highlight.jsにはコンソール出力のための言語識別子が存在しない上、`console`は`shell`のエイリアスに充てられているが、妥協する。

### Line breaking

(スペース2つと改行2階のどちらが好ましいか、調整中...)

### Link

マークダウンのリンクで、リンクテキスト内で`-`や`(`などをエスケープするかどうかは任意とする。

```markdown
🙆‍♂️ [xhiroga/til: What @xhiroga learned with executable code.](https://github.com/xhiroga/til)
🙆‍♂️ [xhiroga/til: What @xhiroga learned with executable code\.](https://github.com/xhiroga/til)
```

[GitHub](https://docs.github.com/ja/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)や[markdownlint](https://github.com/markdownlint/markdownlint/blob/main/docs/RULES.md)にはエスケープのルールはない。

リンクテキストのエスケープは[CreateLinkのローカルルール](https://github.com/ku/CreateLink/blob/3e3c9e6e21178c8d69ed40058fbe25932c14f13f/src/createlink.ts#LL39C30-L39C30)のようだ。

### Unordered List

`-` を用いる。  
[Daring Fireball: Markdown Syntax Documentation](https://daringfireball.net/projects/markdown/syntax#list)によれば `*` , `+` , `-` のいずれも可だが、 `-` は日本語配列・US 配列のいずれでも Shift 無しで入力できるため。

## Meta Rules

### Linter & Formatter

- [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
  - 拡張仕様である[アラート記法](https://docs.github.com/ja/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#alerts)のプレビューが可能
  - テーブルの整形が可能
- [Markdown Footnotes](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-footnotes)
  - Markdown All in OneはFootnotesに対応していないため

## References

開発者 John Gruber の Web サイトを公式リファレンスとみなす。  
[Daring Fireball: Markdown Syntax Documentation](https://daringfireball.net/projects/markdown/syntax)
