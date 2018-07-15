# Git
リーナス・トーバルズによって開発されたバージョン管理ツール。  
`git help merge`などでヘルプ。  

# Usage
## mergeとrebase
複数のコミットを一つにまとめる点では同じであり、どちらを使うかは好みの問題。  
rebaseとインタラクティブrebaseを混同しがち。  
`git rebase <どこにつけ替える> <どのブランチを>`  
`git rebase -i <どのブランチを>` (つけ替える対象はHEAD)  

ポイント: インタラクティブrebaseで順序を修正するのは、ローカルコミットに限ること。リモートとコミット順が食い違うのを防ぐため。  

## resetとrevert
指定したコミットを打ち消すrevertに対して、resetでは指定した場所までHEADを戻す。  
`git reset <戻りたいコミット>`  

ポイント: resetはローカルコミットに限ること。  

## submodule
他のリポジトリの特定のコミットを、自分のリポジトリのサブディレクトリとして登録・参照する。  
```Console
git submodule add https://github.com/hiroga-cc/til # cloneに相当
cd til # サブモジュールはリンクではないため、ディレクトリ・ファイルが存在する
git pull origin master
```

## ログイン情報の省略
`git config credential.helper` とすることで、gitcredentialsによってusernameとpasswordを提供させられる。  
```
git config credential.helper store # store >> ~/.git-credentialsに保存する。
git config credential.helper osxkeychain # macOSの場合はKeyChainを参照することが可能。
```

# 参考
[わかばちゃんと学ぶ Git使い方入門](https://www.amazon.co.jp/dp/B071D4D6XX)
[Learn Git Branching](https://learngitbranching.js.org/)