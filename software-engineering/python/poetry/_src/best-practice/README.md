# Poetryベストプラクティス

## TL;DR

このプロジェクトの`README.md`と`poetry.lock`以外をコピペし、Pythonのバージョンを修正して使ってください。

## プロジェクト作成

```shell
gibo dump python > .gitignore

PYTHON_VERSION=$PYTHON_VERSION
pyenv local $PYTHON_VERSION
poetry config virtualenvs.in-project true --local
poetry init --python $PYTHON_VERSION
poetry add --dev black isort flake8 flake8-bugbear mypy
# リポジトリ作成時は `pre-commit` も追加するとよさそう
poetry install
# `Cmd + Shift + P` で コマンドパレットを開き、 `python.setInterpreter` を実行し、 `.venv` を選択してください。
# isortはワークスペースのPythonから実行されます。

# Jupyter Notebookを利用する場合
poetry add --dev ipykernel
```

- pyenvとpoetryでPythonのバージョン指定方法が異なる。
  - pyenvは`3.10`のような指定が可能。
  - poetryで`3.10`のようにパッチバージョンを指定しない場合、install時にエラーが発生する。`^3.10`ならOK。
- プロジェクト内のスクリプトを、毎回`pyenv shell`や`poetry shell`したくない。
  - VSCodeの設定でIntegrated Terminal起動時にプロジェクトの仮想環境を立ち上げることはできるが、モノリポでプロジェクトごとに仮想環境が異なる場合は不便。
  - `tool.poetry.scripts`は本来タスクランナーではないので、使用を控える。
  - 次の選択肢は `.venv/bin/...`を直接実行すること。この時、絶対パスが入っていると厄介。
  - したがってプロジェクト内に仮想環境を設ける。
    - 念のため、プロファイルに`export POETRY_VIRTUALENVS_IN_PROJECT=true`を設定しておく。
- 1行79文字制限を撤廃したいが、`black`, `flake8`, `isort`のそれぞれに設定が必要なのでおとなしく従う。

## References

- [poetry init では 3.10 が使えるのに、 poetry env use python3.10 ではエラーになる](https://zenn.dev/hiroga/articles/poetry-env-cannot-use-python3_10)
- [Configuration | Documentation | Poetry - Python dependency management and packaging made easy](https://python-poetry.org/docs/configuration/)
- [Python Tips： Poetry の tips あれこれ - Life with Python](https://www.lifewithpython.com/2020/01/python-tips-poetry-tasks.html)
- [Poetry の virtualenv を VSCode に認識させる](https://zenn.dev/takanori_is/articles/let-poetry-create-virtualenv-under-project-folder)
