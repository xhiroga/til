# Git

## submodule
他のリポジトリの特定のコミットを、自分のリポジトリのサブディレクトリとして登録・参照する。  
```Console
git submodule add https://github.com/hiroga-cc/til # cloneに相当
cd til # サブモジュールはリンクではないため、ディレクトリ・ファイルが存在する
git pull origin master
```