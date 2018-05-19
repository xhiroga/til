# iOS


# App
## アプリ起動の流れ
1. UIApplicationMain関数がapplication objectを生成する。このオブジェクトがアプリのライフサイクルに責任を持つ  
2. AppDelegateクラスをapplication objectに代入し、ライフサイクル毎のイベントをロードする  

## ビューについて  
UIKitクラスのcomponentを用いてビューを作成する  
クラス内に`viewDidLoad()`などのイベントを持つ  

## StoryBoard
矢印が発生しているものをScene, それ以外の部品をViewという  