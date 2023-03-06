# OpenAI

## Whisper

```shell
curl -L "https://drive.google.com/uc?export=download&id=1sDAU2ZVI8kiIvLAJGFG9kJmnLsKGs13w" -o 2023_03_07.m4a

export $(cat .env)
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-1" \
  -F file="@2023_03_07.m4a"
# {"text":"おはようございます。今日は3月7日。天気は晴れ、最高気温は19度です。"}
```
