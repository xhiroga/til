# Cookie

TCPコネクション→リクエスト→レスポンスの一連の流れをセッションという。  
ステートレスなHTTPを使って、同じクライアントからの複数のレスポンスに状態を持たせるためにCookieを用いる。

## Development

Command Palette: `Dev Containers: Reopen in Container`

## Run

```shell
poetry shell
python app.py
# Go http://localhost:8888
```

## Cookieを使う流れ

1. サーバーはResponse Headerに"次からCookieに入れて欲しいもの"を指定する(Set-Cookie)
2. クライアントはホスト別にCookieを保存し、次のレスポンスからCookieを含める。
    * ちなみに、一度保存したCookieは次からSet-Cookieしなくてもブラウザが入れてくれる（Firefoxで確認済）
