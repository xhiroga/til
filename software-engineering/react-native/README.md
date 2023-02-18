# React Native

Reactの文法でネイティブアプリを作ることができるFacebook製のOSS。  


# Usage
## react-native initの場合
```console:
react-native init todolist # アプリの雛形を作成
react-native run-ios
```

## CRNA(creat-react-nativa-app)の場合
```console
create-react-native-app myapp # yarn.lockまで含めて初回に作成される
yarn start # Expoのアプリケーションが動作
```


# How it works?
Reactが仮想DOMを使って物理DOMを生成しているように、ネイティブのコンポーネントを生成している。


# TIPS
## react-native init vs create-react-native-app
### react-native init
* メリット
    - Objective-C/ Javaのモジュールを利用できる
* デメリット
    - 起動にXCodeもしくはAndroid Studioが必要
    - macがないとiOSアプリが作成できない
    - スマホをPCにUSB接続しないと実機テストができない
    - フォントのインポートのためにいちいちXCodeを立ち上げる必要がある
    - etc...

### create-react-native-app
* メリット
    - 初期設定が簡単
    - QRコードでアプリを共有できる
    - アプリの実行のためにビルドが不要

* デメリット
    - Objective-C/ Javaのネイティブモジュールを使えない/使ったライブラリをimportできない(注: react-native-cameraなど)
    - ただのHello Worldアプリでも25MBを使う
    - npm run ejectが.gitignoreに対応していない
[Difference between react-native-init and create-react-native-app](https://github.com/react-community/create-react-native-app/issues/516)

## Other Tips
* `console.log()`などでターミナルにObjectを渡すとアプリがフリーズすることがある(Chrome Debuggerなら大丈夫)  
* Viewのサイズは内側のcomponentsのサイズに依存するため、何もないViewだとonPressする場所さえも発生しない  


# TroubleShooting🐯
## React-Native Version Mismatch
JavaScript Version(?)とNative version(package.jsonのバージョン)が異なり、アプリ上のJavaScriptランタイムがReact Nativeを起動できないことがある。使っているターミナルを閉じると治ることがある。


# 参考
The Complete React Native and Redux Course  
https://www.udemy.com/the-complete-react-native-and-redux-course/learn/v4/t/lecture/5738524