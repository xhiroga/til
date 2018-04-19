# Strutsサンプルプロジェクト

Struts1.xのアプリケーションをGradleでビルドしてDocker+Tomcatで動かすサンプル。  

```Console
gragradle build -p hello
# 本当は gradle warを試したかったが、プロジェクト構成とwarプラグインの設定が必要になる  
docker build -t struts-hello .
docker run -ip 18080:8080 struts-hello
curl http://localhost:18080/hello/pages/Who.jsp
```

# 課題
* Strutsのページ構成が正しいか不明。pagesに置いたjspファイルにアクセスさせており、内部でforwardをさせていない。
* Tomcatへのインストール時にwarファイル化せずdocBaseにそのまま置いている。

# 参考
[Strutsの常識を知り、EclipseとTomcatの環境構築 (1/4)](http://www.atmarkit.co.jp/ait/articles/0807/31/news129.html)