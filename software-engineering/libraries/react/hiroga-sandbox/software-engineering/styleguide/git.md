# Git

## Style Rules

### Commit message

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)の形式に従いつつ、命令形（小文字始まり）で変更の持つ効果を説明する。  
[コミットメッセージガイド](https://github.com/RomuloOliveira/commit-messages-guide/blob/master/README_ja-JP.md)は大文字を勧めているが、Shiftの入力が面倒なので避ける。


### Archive Repository

リポジトリをアーカイブする場合、その前に `README.md` に以下の記述を追加する。  
後からリポジトリを発見したメンバーが経緯を把握するのに役立つ。

```md
# Deprecated

[![No Maintenance Intended](https://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

<!-- アーカイブに至った理由と、代替手段 -->

---
```

### Pull Request

- Squash Mergeをするチームの場合、タイトルをコミットメッセージの形式にすること。
- GitHubではPull RequestのタイトルがSquash時のデフォルトコミットメッセージになるため

## References

- [コミットメッセージガイド](https://github.com/RomuloOliveira/commit-messages-guide/blob/master/README_ja-JP.md)
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

