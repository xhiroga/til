# Yarn
npm同様にpackage.jsonを使用可能なJavaScriptのパッケージマネージャー


# Usage
```console
yarn init # package.jsonを作成
yarn add express # package.jsonへの追加/ node_modulesのインストール/ yarn.lockの更新

yarn install # git cloneしたリポジトリ等の依存関係のインストール
```

# TIPS
## yarn.lockについて
package.jsonでバージョン要件を定義しているのとは対照的に、インストールで使用したバージョンを個別に指定している。  
gitへのcommit対象にも含まれるべきとされる。


# Refernce
[yarn - ドキュメント](https://yarnpkg.com/ja/docs)