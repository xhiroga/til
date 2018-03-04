# FlaskでのStaticファイルの扱い

そもそも画像ファイルとかJavaScriptファイルは、htmlファイルをロードした後にサーバーから別にリクエストしている。  
(ネットワークモニターを参照)  
Flaskでは静的ファイルをダウンロードするためのendpointを別途設けており、それがstaticである。  

したがって、呼び出し側でstaticを省略してはいけない（=Flaskはルート直下に静的ファイルを置いているわけではない）  

# 参考
http://flask.pocoo.org/docs/0.10/quickstart/#static-files
