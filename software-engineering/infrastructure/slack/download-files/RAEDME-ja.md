# Download files

結論から言って、 [`files:read`](https://api.slack.com/scopes/files:read) 権限のある [Bot User OAuth Token](https://api.slack.com/authentication/token-types#bot) ではファイルのダウンロードはできなかった。Getリクエストの結果として、`<script>`タグで構成されたHTMLが返却される。

## Run

###  wget

```shell
source .env
wget -d --header="Authorization: Bearer $TOKEN" $URL
```

### Node.js

```shell
pnpm install
node -r @swc-node/register main.ts > out.html
```

## File URLについて

おそらく以下の手順で生成できる。

- `ファイルのメニュー > その他 > ファイルのリンクをコピー` でリンクを取得
- `.env` の `FILE_LINK` を設定
- node -r @swc-node/register get-url-private-download.ts
