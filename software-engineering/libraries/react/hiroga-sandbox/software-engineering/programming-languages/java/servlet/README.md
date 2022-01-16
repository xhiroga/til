# サーブレット

Sunが発表したJavaのAPI。  
静的なドキュメントを返すWebサーバーに対しWebアプリケーションでは動的なドキュメントを返すが、
CGIではインターネットのアクセス数増加に耐えることができず、代わってJavaのアプレットをサーバー側で動作させることでWebアプリケーションとした。  

# デモ
```Console
startApp.sh
curl http://localhost:18080/testbbs/ShowBBS
```

# 概要
## サーブレットのクラスとURLのルーティング
Tomcatの場合、webapp配下のアプリケーションフォルダまたはTomcat/conf/server.xmlで指定したアプリケーションフォルダの
{APP}/WEB-INF/web.xml を参照し、servlet-mappingタグでurlパターンとサーブレット名を、servletタグでservlet名とservletクラスを紐つける。  
Tomcat実行時にインスタンス化されるクラスはWEB-INF配下のclassesフォルダとなる。 
※ 静的コンテンツはルーティングの必要なし(ex. webapp/hello/pages/hello.jsp)  
※ web.xmlに誤字があるとアプリケーションフォルダ以下全て動かなくなるので注意  

### メモ: サーブレットコンテナのためのフォルダ構成
アプリケーションフォルダの配下に静的コンテンツとメタ情報(WEB-INFなど)を並列に配置する。  
開発時のwebapp配下がそのままパッケージングされ、かつclassファイルがWEB-INF配下に収まるイメージ。  
mvn paclageで自動的にやってくれる。
```
tomcat/webapp/testbbs
+--index.jsp
+--WEB-INF # このフォルダ名にするとWebサーバーが外部に公開しない約束になっている
   +--classes # したがってバイナリファイルを置いておくのに都合が良い
   +--cc/hiroga/ShowBBS.class
   web.xml
+--META-INF
...
```



# 参考
[JSP and Servlet Overview](http://www.pearsonitcertification.com/articles/article.aspx?p=29786)