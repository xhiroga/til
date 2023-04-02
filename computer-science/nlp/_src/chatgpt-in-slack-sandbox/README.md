# ChatGPT in Slack

```shell
op inject --force -i .env.tpl -o .env

poetry install
poetry shell
dotenv run -- python ./ChatGPT-in-Slack/main.py
```
