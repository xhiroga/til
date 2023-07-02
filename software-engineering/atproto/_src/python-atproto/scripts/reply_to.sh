#!/bin/zsh
set -euo pipefail  # -e: エラー時のスクリプト終了, -u: 未設定変数の参照エラー, -o pipefail: パイプラインエラー時の終了

export $(cat .env | grep -v ^# | xargs)

response=$(curl -s -X POST "https://bsky.social/xrpc/com.atproto.server.createSession"\
  -H 'Content-Type: application/json' \
  -d "{\"identifier\": \"$HANDLE\", \"password\": \"$PASSWORD\"}" \
)
access_jwt=$(echo $response | jq -r .accessJwt)

text="The first email was sent by Ray Tomlinson in 1971. He introduced the '@' symbol in email addresses. The first spam email was sent by Gary Thuerk in 1978."
createdAt=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

json_data="{
  \"collection\": \"app.bsky.feed.post\",
  \"record\": {
    \"createdAt\": \"$createdAt\",
    \"text\": \"@hiroga.bsky.social $text\",
    \"reply\": {
      \"root\": {
        \"cid\": \"$CID\",
        \"uri\": \"$URI\"
      },
      \"parent\": {
        \"cid\": \"$CID\",
        \"uri\": \"$URI\"
      }
    },
    \"\$type\": \"app.bsky.feed.post\"
  },
  \"repo\": \"$DID\"
}"

echo "$json_data"

curl -vX POST "https://bsky.social/xrpc/com.atproto.repo.createRecord" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $access_jwt" \
-d "$json_data"
