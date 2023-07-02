#!/bin/zsh
set -euo pipefail  # -e: エラー時のスクリプト終了, -u: 未設定変数の参照エラー, -o pipefail: パイプラインエラー時の終了

export $(cat .env | grep -v ^# | xargs)

curl --http1.1 -vX GET 'https://bsky.social/xrpc/app.bsky.feed.getTimeline?algorithm=reverse-chronological' \
-H "Authorization: Bearer $TOKEN" \
-H 'Content-Type: application/json'
