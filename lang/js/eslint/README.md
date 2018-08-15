# ESLint

JavaScriptのためのlint(静的解析)ユーティリティ。

## 使い方
### コマンドラインで使う
※事前に.eslintrc.jsonファイルを作成済みであること。
```console:
npm install --save-dev eslint # まずはeslintをdevDependenciesへの登録付きでインストール。
npm install -g eslint-cli # eslintコマンドからローカルのeslintが使えるようにしてくれるパッケージ。
eslint test.js
```

### Atomで使う
以下の2点でOK。  
1. .eslintrc ファイルを作成済みであること
2. linter-eslint パッケージをインストールしていること

## 参考
ESLint をグローバルにインストールせずに使う  
https://qiita.com/mysticatea/items/6bd56ff691d3a1577321
