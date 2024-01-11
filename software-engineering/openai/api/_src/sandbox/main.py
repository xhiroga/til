import openai
import os
import sys
from dotenv import load_dotenv


def main():
    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "日本語で返答してください。"
            },
            {
                "role": "user",
                "content": "What is AI?"
            },
        ],
    )
    for choice in res["choices"]:
        message = choice["message"]["content"].encode(
            'utf-8')
        sys.stdout.buffer.write(message)


if __name__ == "__main__":
    main()

## References
# [【最新速報】ChatGPT APIの「概要と使い方」（Pythonコード付き）](https://zenn.dev/umi_mori/articles/chatgpt-api-python)
