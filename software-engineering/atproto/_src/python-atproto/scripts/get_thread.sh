#!/bin/zsh
set -euo pipefail  # -e: エラー時のスクリプト終了, -u: 未設定変数の参照エラー, -o pipefail: パイプラインエラー時の終了

export $(cat .env | grep -v ^# | xargs)

response=$(curl -s POST "https://bsky.social/xrpc/com.atproto.server.createSession"\
  -H 'Content-Type: application/json' \
  -d "{\"identifier\": \"$HANDLE\", \"password\": \"$PASSWORD\"}" \
)
access_jwt=$(echo $response | jq -r .accessJwt)

echo "\n--- root ---"
curl --http1.1 -vX GET 'https://bsky.social/xrpc/app.bsky.feed.getPostThread?uri=at://did:plc:et47te5fb7uv64pbltu37lcc/app.bsky.feed.post/3jzjhulhuys2r' \
-H "Authorization: Bearer $access_jwt" \
-H 'Content-Type: application/json'

echo "\n--- 2nd ---"
curl --http1.1 -vX GET 'https://bsky.social/xrpc/app.bsky.feed.getPostThread?uri=at://did:plc:d7mnkzaznaop33oiowcbco7g/app.bsky.feed.post/3jzjhxjagl72v' \
-H "Authorization: Bearer $access_jwt" \
-H 'Content-Type: application/json'

echo "\n--- 3rd ---"
curl --http1.1 -vX GET 'https://bsky.social/xrpc/app.bsky.feed.getPostThread?uri=at://did:plc:et47te5fb7uv64pbltu37lcc/app.bsky.feed.post/3jzjhzotcdp2v' \
-H "Authorization: Bearer $access_jwt" \
-H 'Content-Type: application/json'
