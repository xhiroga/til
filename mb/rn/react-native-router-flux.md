# react-native-router-flux

# iconについて
iconプロパティは、class/functionを引数にとるTabBarIcon()を呼び出す。  
その際、propsとしてselected, title, iconNameを受け取り可能。

# TroubleShooting
* null is not an object AppNavigator.router
Routerの直下はcomponentを含まないSceneタグである必要がある。
