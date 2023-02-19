# Poetry の virtualenv を VSCode に認識させる

```bash
poetry config virtualenvs.in-project true --local
# OR
export POETRY_VIRTUALENVS_IN_PROJECT=true

pyenv local 3.10.1
poetry init --python ^3.10
poetry install # installの段階で.venvが作成される
```


## References

- [poetry init では 3\.10 が使えるのに、 poetry env use python3\.10 ではエラーになる](https://zenn.dev/hiroga/articles/poetry-env-cannot-use-python3_10)
- [Configuration \| Documentation \| Poetry \- Python dependency management and packaging made easy](https://python-poetry.org/docs/configuration/)
- [Python Tips： Poetry の tips あれこれ \- Life with Python](https://www.lifewithpython.com/2020/01/python-tips-poetry-tasks.html)
- [Poetry の virtualenv を VSCode に認識させる](https://zenn.dev/takanori_is/articles/let-poetry-create-virtualenv-under-project-folder)
