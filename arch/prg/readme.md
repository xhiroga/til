# PRGパターン    
Post-Redirect-Getパターン.    
Post後にGetリクエストを送信させることで、再読込による二重送信を防ぐ.    
ブラウザバックを防ぐことはできないので注意.    
    
    
# 動かし方    
* ローカルの場合    
$ python app.py    
    
* herokuの場合    
$ # リポジトリを紐つける等    
$ git push heroku master    
    
localhost:5000 にアクセスし、ネットワークの監視画面を開いて画面を操作するとリダイレクトしているのが分かる.    
  

