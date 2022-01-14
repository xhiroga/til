# Heroku

ログ確認  
$ heroku logs -t --app APP_NAME  

アプリ再起動  
$ heroku restart --app APP_NAME  

ローカル起動  
$ heroku local web(Procfileのプロセス名)  

環境変数の設定  
$ heroku config:set URL="foo://baa" --app APP_NAME  

# 設定
* Procfile
プロセス名: コマンド の順に記述する。
ex) web: gunicorn gettingstarted.wsgi --log-file -
webという名前のプロセスは自動的にHTTP routingに紐付けられる。
複数プロセスの定義も可能。

* Pipfile
pipenvの設定ファイル。

# その他
hobbyプランだと30分でアプリがsleepする。professionalだと寝ない+scale可能。
