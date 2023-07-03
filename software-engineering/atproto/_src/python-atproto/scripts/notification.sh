#!/bin/zsh
set -euo pipefail  # -e: エラー時のスクリプト終了, -u: 未設定変数の参照エラー, -o pipefail: パイプラインエラー時の終了

export $(cat .env | grep -v ^# | xargs)

response=$(curl -s POST "https://bsky.social/xrpc/com.atproto.server.createSession"\
  -H 'Content-Type: application/json' \
  -d "{\"identifier\": \"$HANDLE\", \"password\": \"$PASSWORD\"}" \
)
access_jwt=$(echo $response | jq -r .accessJwt)

curl --http1.1 -vX GET 'https://bsky.social/xrpc/app.bsky.notification.listNotifications' \
-H "Authorization: Bearer $access_jwt" \
-H 'Content-Type: application/json'
