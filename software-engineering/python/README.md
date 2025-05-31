# Python

## モジュール・パッケージ

実験は[`_src/module`](_src/module)を参照。

Pythonにおいて、`.py` ファイルをモジュール、複数のモジュールをまとめたものをパッケージと呼ぶ。

同じパッケージという名前で、次のような異なった概念がある。混乱のもとになっている。

- 配布パッケージ: `pip install $PACKAGE_NAME` で指定するパッケージ
- インポートパッケージ: import $PACKAGE_NAME で指定するパッケージ

例えば`Pillow`の場合、配布パッケージは`Pillow`であり、インポートパッケージは`PIL`である。

### install

`pip install $PACKAGE_NAME` でパッケージをインストールすると、その配布パッケージに対応するインポートパッケージが`site-packages`にインストールされる。

単一の配布パッケージが複数のインポートパッケージを持つこともある。

```console
$ uv add attrs
$ uv run python -c "import attr"
$ uv run python -c "import attrs"
```

### import

`sys.path`以下のパスに含まれるモジュール・パッケージをimportできる。

```console
$ .venv/bin/python -c "import sys; print(sys.path)"
# OR
$ uv run python -c "import sys; print(sys.path)"

['', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python313.zip', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13/lib-dynload', '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/.venv/lib/python3.13/site-packages', '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src']
```

### ビルドと配布

Pythonのソースコードの配布は、初期にはソースコードを直接共有する形で行われた。その後、リポジトリから配布用のソースコードをビルドする手順として`setup.py`が導入された。

ビルド手順の導入に伴い、GitHubなどのVCSからパッケージを直接ダウンロードすることが可能になった。また、ビルドツールの多様化や静的解析ツールからの要望を受けて`setup.py`の代わりに静的設定ファイル（`setup.cfg`, `pyproject.toml`）が導入される。


