# Python

## モジュール・パッケージ

実験は[`_src/module`](_src/module)を参照。

Pythonにおいて、`.py` ファイルをモジュール、モジュールの含まれたディレクトリをパッケージと呼ぶ。

同じパッケージという名前で、次のような異なった概念がある。混乱のもとになっている。

- 配布パッケージ: `pip install $PACKAGE_NAME` で指定するパッケージ
- インポートパッケージ: `import $PACKAGE_NAME` で指定するパッケージ

例えば`Pillow`の場合、配布パッケージは`Pillow`であり、インポートパッケージは`PIL`である。

インポートパッケージには2種類ある。Python3.2以前から利用できた`regular package`と、Python3.3から利用できる`namespace package`である。

（なお、`regular package`という用語は[PEP420](https://peps.python.org/pep-0420/)から引用した）

```console
% uv run python
>>> from module import spam, ham
>>> spam
<module 'module.spam' from '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src/module/spam/__init__.py'>
>>> ham
<module 'module.ham' (namespace) from ['/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src/module/ham']>
```

### install

`pip install $PACKAGE_NAME` でパッケージをインストールすると、その配布パッケージに対応するインポートパッケージが`site-packages`にインストールされる。

単一の配布パッケージが複数のインポートパッケージを持つこともある。

```console
$ uv add attrs
$ uv run python -c "import attr"
$ uv run python -c "import attrs"
```

配布パッケージを用いずにインポートパッケージを追加する方法もある。`.whl`ファイルや`GitHub`のリポジトリを直接指定すればよい。

```console
$ make somepackage/dist/somepackage-0.1.0-py3-none-any.whl
$ uv add somepackage/dist/somepackage-0.1.0-py3-none-any.whl
```

### import

Pythonは、パッケージ・モジュールを配置すべきパスをモジュール検索パスとして提供している。モジュール検索パスは`site`や`sys.path`で確認できる。

```console
$ .venv/bin/python -m site
$ uv run python -m site

sys.path = [
    '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module',
    '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python313.zip',
    '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13',
    '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13/lib-dynload',
    '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/.venv/lib/python3.13/site-packages',
    '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src',
]
USER_BASE: '/home/hiroga/.local' (exists)
USER_SITE: '/home/hiroga/.local/lib/python3.13/site-packages' (doesn't exist)
ENABLE_USER_SITE: False

$ .venv/bin/python -c "import sys; print(sys.path)"
$ uv run python -c "import sys; print(sys.path)"

['', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python313.zip', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13/lib-dynload', '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/.venv/lib/python3.13/site-packages', '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src']
```

モジュール検索パスは、とりあえず次のとおり構成される。

1. カレントディレクトリ
2. `PYTHONPATH`環境変数で指定したパス
3. `site-packages`ディレクトリ
4. `site-packages`以下の`.pth`ファイルで指定したパス

`.pth`ファイルは、歴史的には`PYTHONPATH`の代替手段として登場したらしい。1行に1つのパスを含むシンプルなファイルである。

```console
$ cat .venv/lib/python3.13/site-packages/_module.pth
/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src
```

### スクリプト実行・モジュール実行

[StackOverFlowの回答](https://stackoverflow.com/questions/7610001/what-is-the-purpose-of-the-m-switch)も参照。

`python -m main.py` のようなモジュール実行は、Python2.4.1で登場した。

Pythonのユースケースが広がるにつれて、WebアプリケーションフレームワークやCLIツールなど、ライブラリがブートローダー部分を担うケースが登場した。その際もモジュールのimport時と同様に正確なパスを知らなくても使いたいという要望が生まれたのだろう。そうした背景から`-m`オプションが導入された。

### ビルドと配布

Pythonのソースコードの配布は、初期にはソースコードを直接共有する形で行われた。その後、リポジトリから配布用のソースコードをビルドする手順として`setup.py`が導入された。

ビルド手順の導入に伴い、GitHubなどのVCSからパッケージを直接ダウンロードすることが可能になった。また、ビルドツールの多様化や静的解析ツールからの要望を受けて`setup.py`の代わりに静的設定ファイル（`setup.cfg`, `pyproject.toml`）が導入される。

#### 

### モジュール・パッケージのトラブルシューティング

#### なぜかパッケージをインポートできた

初めに、元になったモジュール検索パスを確認しましょう。

```console
$ uv run python -c "import importlib, pkgutil; [print(importlib.util.find_spec(mod.name)) for mod in pkgutil.iter_modules()]"
```

次に、そのパスがなぜモジュール検索パスに含まれているかを確認しましょう。

これは便利な方法が発見できなかったので、[### モジュール検索パスの構成](#import)を参照して手動で切り分けます。

#### 相対インポートができない

#### 配布するパッケージにおいて、絶対インポートと相対インポートのどちらを採用すべきか

#### srcレイアウトを採用したら、パッケージのインポートが上手くいかない

私が実際に`[blender-mcp-senpai](https://github.com/xhiroga/blender-mcp-senpai)`で遭遇したケース。srcレイアウトを採用した上で、`uv run python src/blender-mcp-senpai/main.py`のように実行すると、`ImportError`が発生しました。

