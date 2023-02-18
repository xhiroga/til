# Swagger
RESTful API作成支援のツール群。  
* YAML/JSONで作成可能な仕様ドキュメントとそのエディター。  
* リクエストの発行まで可能なドキュメント生成  
* サーバー/クライアント両方のコード生成（サーバーはモック）  
* API仕様をアップロードするとAWSでAPI Gatewayを作成可能。  


# Swagger Spec
現在はOpenAPI仕様。  
https://swagger.io/specification/  

大まかな構成は...  
1. swagger(バージョン), info(APIの解説), paths(パス)の3つのプロパティが必須。  
2. pathsの内側には操作対象(ex. pet)と操作(ex. get, post...)がある。  
3. 操作の内側にparameterとresponseの定義がある。  


# Swagger Codegen
OpenAPI仕様を元に、選択した言語でサーバーのスタブとクライアントのSDKを作成する。  
開発初期段階などにクライアントサイドがAPIモックを利用して開発を進めることができる。  


# 参考
Swagger Petstore: ペットストアのサーバーのサンプル
http://petstore.swagger.io/v2/swagger.json
