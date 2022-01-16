# Navigator
ページ遷移を担当するライブラリ。


# react-navigation
React Native公式のnavigation用ライブラリ。


# react-native-router-flux
React Nativeで画面遷移を実現するためのライブラリ。3rd Partyだがシンプルさが魅力。  

Routerコンポーネント...複数のSceneコンポーネントを内包するコンポーネント。  
Sceneコンポーネント...key: componentの組み合わせでそれぞれの画面を持つ。  
Action API...Sceneで定義したkeyを指定componentへ移動する関数として持つAPI。onPressなどに組み込んで使用する。  

# Usage


※ ModalとTabsを併用すると、Tabs自身もnavbarを表示するためnavbarが2段階になってしまう。  
TabsにhideNavBar属性を明記することで防ぐことが可能。  


# Reference
https://github.com/aksonov/react-native-router-flux