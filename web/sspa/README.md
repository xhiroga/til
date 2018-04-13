# サーバレスシングルページアプリケーション

はじめにWebアプリケーションの静的Webホスト(S3)へのアップロード、ルーティングとビューの設定を実施。  
ついでDynamoDB, Lmabda, Cognitoを設定する。  

アプリケーションロジックはブラウザ側に、しかしセキュリティのための検証はサーバー側に。

# 構成
<head>でCSS/JavaScriptを読み込む。  
CSS...ブラウザ間の差異を無くす"normalize.min.css", ミニマムなレスポンシブデザインの"skeleton.min.css", およびSkeletonのフォントをGoogleFontから取得する。  
JavaScript...DynamoDBなどとの通信モジュールを"vendor.js"に濃縮。自分のスクリプトを"app.js"に作成。  

## S3へのデプロイ
```
$ aws s3 sync public/ s3://hiroga.sspa.learnjs --acl public-read  
```

## ランディングページ
サービスの説明とCTA(Call to Action)ボタンのあるページ。サンプルを流用するのが早い。  
[http://getskeleton.com/examples/landing/](http://getskeleton.com/examples/landing/)  

## ハッシュイベント
JavaScriptでURLハッシュ(ex...#customer)の変更を監視するルータ関数(=コントローラ)からビュー関数を変更する。  
この段階からテスト駆動で作成する。また、作成したテストは本番環境にデプロイする(不具合の調査に使用するため)  

## テスト
Jasmineを利用する。"jasmin.js"と"app.js"および"app_spec.js"を組み込んだテスト用のHTMLファイルをブラウザで開くことでテストを実行する。  
テストコードはスイーツ/スペック(JUnitでいうクラス/メソッド)で構成される。  
ルータ関数のない状態で書かれるはずなので、名前空間→ルータ関数→アサーション実行、の順にエラーを取り除いていくこと。  

## ページのロード
DOMをロードしてからJavaScriptを読み込むように、2つの対策をとる。  
1. $.ready() でページのローディングに反応させる  
2. body要素の最後でJavaScriptを評価する  

# 参考
[サーバレスシングルページアプリケーション(O'REILLY)](https://www.oreilly.co.jp/books/9784873118062/)