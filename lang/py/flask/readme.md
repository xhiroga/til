# コード
1. Flaskオブジェクトのインスタンス化
2. appデコレータでURLごとに呼ばれるメソッドの定義
3. run実行（必要に応じて）

# 実行方法
1. メソッド
app.run(host='localhost', port=5000)
内部のWerkzeug(ヴェルクツォイク)サーバーを起動してくれるが、プロダクション環境ではWSGI準拠サーバーを使うこと。

2. CLI
$ export FLASK_APP=app.py
$ flask run


# 実践
* 画像ファイル, JavaScriptファイルの取得  
静的ファイルは全てapp.py と同じ階層のstaticフォルダに格納する。  

* URLクエリパラメータの取得  
ex) http://127.0.0.1:5000/?speed=60  
request.args.get('speed') # 60  


# その他
debugフラグがオンだとホットスワップする(app.debug=True)  
その用途ならrun()ではなくflaskCLIを使うべし。  
