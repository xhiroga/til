

# MEMO
node-fetchだと、userpoolへのサインインまではいけるがidpoolに繋ぐところでエラーが発生する。
browserifyを利用してブラウザのjavascriptから実行するのが良い。
```
{
  "code": "UnknownError",
  "message": "Unknown error, the response body from fetch is: undefined"
}
```