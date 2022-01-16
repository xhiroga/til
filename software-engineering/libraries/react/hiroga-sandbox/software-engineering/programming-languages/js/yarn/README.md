# Yarn
npm同様にpackage.jsonを使用可能なJavaScriptのパッケージマネージャー


# Usage
```console
yarn init # package.jsonを作成
yarn add express # package.jsonへの追加/ node_modulesのインストール/ yarn.lockの更新

yarn install # git cloneしたリポジトリ等の依存関係のインストール
```


# TIPS
## npm install -g <package...> に代わるコマンド
`yarn global add <package...>`で代用できる(ただし、インストール後にコンソールの再起動が必要かも)     
yarnのポリシー的にはグローバルに依存関係を持つべきではない。  
各yarnプロジェクトの`./node_modules/.bin/`にコマンドが格納されているため、そちらが使われるようにパスを通す(+それぞれのyarnプロジェクト内でだけコマンドを使用する)ように努めるべき。  
`export PATH="./node_modules/.bin:$PATH"`

## yarn.lockについて
package.jsonでバージョン要件を定義しているのとは対照的に、インストールで使用したバージョンを個別に指定している。  
gitへのcommit対象にも含まれるべきとされる。


# Refernce
[yarn - ドキュメント](https://yarnpkg.com/ja/docs)