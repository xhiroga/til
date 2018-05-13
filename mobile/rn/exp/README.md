# Expo
React Nativeアプリケーションを開発・共有するためのOSSおよびプラットフォーム。  
Expo XDEとExpo CLIがある。


# Usage
```
exp init
exp start # QRコード経由でスマホのExpoを起動する。
```


# TIPS
## CRNA(craete-react-native-app)と exp init の違い
CRNAはExpo devtoolの一部のみを使用している。例えば、Expoアカウントでのログインは不要。  

## iOSのExpoクライアントへの制限
自身のアカウント以外から公開されているアプリのプレビューができなくなり、代わりに次のようなエラーメッセージが表示される。    
```
Sorry, you are not allowed to load "(PROJECT NAME)"
...
Expo Client can only be used to view your own projects. To view this project, please ensure you are signed in to the same Expo account that created it.
```

## Others
* プロジェクトのapp.jsonに"expo"がなければ、そのアプリはexpo対応ではない。
