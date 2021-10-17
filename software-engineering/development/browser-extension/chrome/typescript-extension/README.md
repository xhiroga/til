# Typescript extension

## Architecture Decision Record

- フォーマットにPrettierを、LintにESLintを用いる。
  - List時にPrettierのルールで上書きするように。
- VSCodeのExtensionでもPrettierとESLintを導入する。

###  Update from [chibat/chrome\-extension\-typescript\-starter](https://github.com/chibat/chrome-extension-typescript-starter)

- Use Yarn
- No `main` in package.json
- No use rimraf
- Simpler webpack config

## References

- [TypeScriptのプロジェクトにESLintとPrettierを導入する方法（\+VSCodeでの設定） \- Qiita](https://qiita.com/yuma-ito-bd/items/cca7490fd7e300bbf169)
- [chibat/chrome\-extension\-typescript\-starter: Chrome Extension TypeScript Starter](https://github.com/chibat/chrome-extension-typescript-starter)
- [TypeScriptで作るイマドキChrome拡張機能開発入門 \- Qiita](https://qiita.com/markey/items/ea9ed18a1a243b39e06e)