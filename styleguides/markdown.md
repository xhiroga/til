# Markdown Style Guide

## Style Rules

### Heading

- URLのパスに用いられることがあるので、簡潔な英語を推奨する。
- 英語の場合、大文字で始める。入力する手間より、見出しが単語ではなく文になったときの小文字始まりの違和感が大きいため。

## Formatting Rules

### Heading

- (`##` と `--` のどちらを用いるか、調整中...)
- (Headingと本文の間に一行入れるか、調整中...)

### Code Block for Shell

(shellとbashのどちらを用いるか、調整中...)

## Link

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

## References

開発者 John Gruber の Web サイトを公式リファレンスとみなす。  
[Daring Fireball: Markdown Syntax Documentation](https://daringfireball.net/projects/markdown/syntax)
