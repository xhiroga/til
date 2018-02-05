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

# その他
debugフラグがオンだとホットスワップする(app.debug=True)  
その用途ならrun()ではなくflaskCLIを使うべし。  
