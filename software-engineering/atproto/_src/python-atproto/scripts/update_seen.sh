#!/bin/zsh
set -euo pipefail  # -e: エラー時のスクリプト終了, -u: 未設定変数の参照エラー, -o pipefail: パイプラインエラー時の終了

export $(cat .env | grep -v ^# | xargs)

response=$(curl -s -X POST "https://bsky.social/xrpc/com.atproto.server.createSession"\
  -H 'Content-Type: application/json' \
  -d "{\"identifier\": \"$HANDLE\", \"password\": \"$PASSWORD\"}" \
)
access_jwt=$(echo $response | jq -r .accessJwt)

seenAt=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
json_data="{\"seenAt\": \"${seenAt}\"}"
echo "$json_data"

curl -vX POST "https://bsky.social/xrpc/app.bsky.notification.updateSeen" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $access_jwt" \
-d "$json_data"
