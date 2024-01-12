# znish

```shell
make
pyenv local 3.10
poetry install -C znish

op inject --force -i .env.tpl -o .env
export $(cat .env)

poetry shell -C znish
python ./znish/znish/slack.py
```

## Note

```terminal
ERROR:slack_bolt.MultiTeamsAuthorization:Although the app should be installed into this workspace, the AuthorizeResult (returned value from authorize) for it was not found.
```

## References

- [laiso/znish: Slack BOT created with LangChain that speaks Japanese](https://github.com/laiso/znish)
