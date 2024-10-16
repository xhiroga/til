# Makefile

makeをタスクランナーとして使用するための、個人的なMakefileのスタイルガイドです。

## Style Rules

### Assignment to variables （変数への代入）

できるだけ := を使用します。= で定義された変数は再帰的に参照されるため、挙動がややトリッキーになるため。
詳細は [6.5 Setting Variables](https://www.gnu.org/software/make/manual/make.html#Setting)) を参照してください。

### Order or targets （ターゲットへの代入）

以下が推奨されるターゲットの順序です。

```markdown
1. all (any name you like)
2. .PHONY
3. (その他のタスク)
4. clean
```

`all` は任意で、`download`、`build`、`start` などのデフォルトタスクを含むべきです。目標を指定せずに実行することを許可しない場合は、このターゲットを削除してください。

`.PHONY` ターゲットには、ビルド出力ではないすべての擬似ターゲットを含める必要があります。擬似ターゲットを使用する理由は2つあります。同名のファイルが存在する場合にターゲットを確実に実行するためと、同名のファイルのタイムスタンプをチェックせずに済むためです。

詳細は [Phony Targets (GNU make)]((<https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html>)) を参照してください。

`clean` は、私が知っている限り、最後に配置されることが推奨されています。

## Formatting Rules

### Command substitution （コマンド置換）

`\`（バッククォート）ではなく、`$$()`（ダブルドル）を使用します。
一部の記事では、バッククォートは古いmakeに対してわずかに互換性が高いと述べていますが、ネストされたコマンド代入でバッククォートをエスケープするのは面倒です。

#### TIPS: `$$(command)` vs `$(shell command)`

`$(shell command)` はMakefileの読み込み時に評価されます。一方、`$$(command)` は実行時に評価されます。詳細は[3.8 How Makefiles Are Parsed](https://www.gnu.org/software/make/manual/make.html#Parsing-Makefiles)と[3.9 Secondary Expansion](https://www.gnu.org/software/make/manual/make.html#Secondary-Expansion)を参照してください。

### Semicolon vs new line and TAB （セミコロン vs 改行とタブ）

コマンドには新しい行とTABを使用します。コマンドの行数に関係なくフォーマットを一定に保つためです。

### Shell variables （シェル変数）

シェル変数とmake変数を区別するために、`$$VAR`の代わりに`$${VAR}`を使用します。

## Meta Rules

### File name （ファイル名）

`Makefile` を使用し、大文字で始めます。
マニュアルには「他の重要なファイル（例：`README`）と同じように、ディレクトリリストの先頭近くに目立つように表示されるため、Makefileを推奨します」と記載されています。詳細は [Makefile-Names](https://www.gnu.org/software/make/manual/make.html#Makefile-Names) を参照してください。

## References and Inspirations

- [makefile styleguide as task runner](https://dev.to/hiroga/makefile-styleguide-as-task-runner-3i75)
  - このStyleguideは私のdev.to記事の日本語訳です。
- [GNU make](https://www.gnu.org/software/make/manual/make.html)
- [Home \| Makefile Advent Calendar 2020](https://voyagegroup.github.io/make-advent-calendar-2020/)
- [Makefile Best Practices — Cloud Posse Developer Hub](https://docs.cloudposse.com/reference/best-practices/make-best-practices/)
- [bash \- What is the difference between $\(command\) and \`command\` in shell programming? \- Stack Overflow](https://stackoverflow.com/questions/4708549/what-is-the-difference-between-command-and-command-in-shell-programming)
- [bash \- Command substitution: backticks or dollar sign / paren enclosed? \- Stack Overflow](https://stackoverflow.com/questions/9405478/command-substitution-backticks-or-dollar-sign-paren-enclosed)
- [coding style \- Should I name "makefile" or "Makefile"? \- Stack Overflow](https://stackoverflow.com/questions/12669367/should-i-name-makefile-or-makefile)
- [Style Guide の Style Guide を作る](https://zenn.dev/hiroga/articles/styleguide-of-styleguide)
