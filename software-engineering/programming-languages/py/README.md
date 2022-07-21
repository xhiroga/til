# Python

## Getting Started(2022)

```shell
mkdir $PROJECT_NAME
cd $PROJECT_NAME
poetry config virtualenvs.in-project true --local
poetry init --python ">=3.9,<3.11"  # 詳しくないがScipyの制約
poetry install
code .
# Select interpreter by `Cmd + Shift + P`
```


# Reference
[Python 3.7とVisual Studio Codeで型チェックが捗る内作Pythonアプリケーション開発環境の構築](https://qiita.com/shibukawa/items/1650724daf117fad6ccd)
