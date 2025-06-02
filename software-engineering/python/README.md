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

## editable install

実験は[`_src/editable-install`](_src/editable-install)を参照。

カレントディレクトリをパッケージとしてinstallすると、`site-packages`ディレクトリを経由してパッケージを利用できる。そのためimport文のモジュール名をパッケージ名から始めることができる。

```console
$ python3 -m venv .venv
$ .venv/bin/pip install -e .
$ .venv/bin/python src/editable_install/main.py
Hello, World!
```

この際、`site-packages`ディレクトリには`.pth`ファイルが作成される。

```console
$ cat .venv/lib/python3.13/site-packages/_editable_install.pth
/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/editable-install/src
```

`pip install -r requirements.txt`と`pip install -e .`の両方を実行するのは手間なので、`requirements-dev.txt`内に`-e .`を記述しておくと楽になる。なぜrequirements.txtではないかというと、パッケージが自分自身を`site-packages`に配置したいような用途では、本番=配布用の依存関係解決に別途`setup.py`や`pyproject.toml`が用意されていることが普通のため。

```console
$ rm -rf .venv
$ python3 -m venv .venv
$ cat requirements.txt
-e .

$ .venv/bin/pip install -r requirements.txt
$ cat .venv/lib/python3.13/site-packages/_editable_install.pth
/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/editable-install/src

$ .venv/bin/python src/editable_install/main.py
Hello, World!
```

例えば、`requests`が同様の構成を取っている。

```console
$ rm -rf .local
$ mkdir -p .local
$ git -C .local clone https://github.com/psf/requests.git
$ cd .local/requests
$ python3 -m venv .venv
$ .venv/bin/pip install -r requirements-dev.txt
$ cat .venv/lib/python3.13/site-packages/__editable__.requests-2.32.3.pth
/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/.local/requests/src
```

なお、`uv`を利用している場合は、`[build-system]`が宣言されているならば`uv sync`の際に自動的に`-e .`を実行する。

```console
$ rm -rf .venv && rm uv.lock
$ uv sync --verbose
DEBUG uv 0.7.3 (Homebrew 2025-05-07)
...
Creating virtual environment at: .venv
...
DEBUG Adding direct dependency: editable-install*
DEBUG Directory source requirement already cached: editable-install==0.1.0 (from file:///home/hiroga/Documents/GitHub/til/software-engineering/python/_src/editable-install)
Installed 1 package in 3ms
 + editable-install==0.1.0 (from file:///home/hiroga/Documents/GitHub/til/software-engineering/python/_src/editable-install)
$ uv run python src/editable_install/main.py
Hello, World!
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

Python3でモジュールをimportするとき、モジュール名の頭に`.`を付けないなら、それは絶対importである。アプリケーション開発のレイアウトではエントリーポイントはプロジェクトルートに置かれるのが通例なので、プロジェクトルートからの相対importのようにも見えるが、異なる。モジュール検索パス内のカレントディレクトリからの絶対importである。

パッケージを配布するとき、そのパッケージがパッケージ管理ツール(pip, uv, etc...)によってインストールされるなら、パッケージは`site-packages`ディレクトリ経由でimportされる。なお、そうではない例としては、アプリケーション拡張機能(Blender, ComfyUI, etc...)など独自の方法でソースコードを管理する場合が考えられる。

したがって、配布用のパッケージでは、カレントディレクトリからの絶対importを採用すると、インストール時に動かなくなることがある。これを避けるため、パッケージ開発においては`Editalbe install`を用いることで`site-packages`ディレクトリから参照するよう統一することがベストプラクティスになっている。

### スクリプト実行・モジュール実行

[StackOverFlowの回答](https://stackoverflow.com/questions/7610001/what-is-the-purpose-of-the-m-switch)も参照。

`python -m main.py` や `python -m http.server` のようなモジュール実行は、Python2.4.1で登場した。

Pythonのユースケースが広がるにつれて、WebアプリケーションフレームワークやCLIツールなど、ライブラリがエントリーポイントを担うケースが登場した。その際もモジュールのimport時と同様に正確なパスを知らなくても使いたいという要望が生まれたのだろう。そうした背景から`-m`オプションが導入された。

モジュール実行の場合、エントリーポイントを除くアプリケーションのコード内でも相対importを用いることが可能になる。パッケージ開発では`Editable install`を用いるのがベストプラクティスであると紹介したが、`-m`オプションでの開発と相対importの組み合わせも広がっていくかもしれない。

なお、`Editable install`を用いる場合、エントリーポイント**以外の**アプリケーションコード内で相対importが可能になる。ややこしいので避けた方が良さそうだ。

### ビルドと配布

Pythonのソースコードの配布は、初期にはソースコードを直接共有する形で行われた。その後、リポジトリから配布用のソースコードをビルドする手順として`setup.py`が導入された。

ビルド手順の導入に伴い、GitHubなどのVCSからパッケージを直接ダウンロードすることが可能になった。また、ビルドツールの多様化や静的解析ツールからの要望を受けて`setup.py`の代わりに静的設定ファイル（`setup.cfg`, `pyproject.toml`）が導入される。

### モジュール・パッケージのトラブルシューティング

#### なぜかパッケージをインポートできた

初めに、元になったモジュール検索パスを確認しましょう。

```console
$ uv run python -c "import importlib, pkgutil; [print(importlib.util.find_spec(mod.name)) for mod in pkgutil.iter_modules()]"
```

次に、そのパスがなぜモジュール検索パスに含まれているかを確認しましょう。

これは便利な方法が発見できなかったので、[モジュール検索パスの構成](#import)を参照して手動で切り分けます。

#### srcレイアウトを採用したら、パッケージのインポートができなくなった

私が実際に`[blender-mcp-senpai](https://github.com/xhiroga/blender-mcp-senpai)`で遭遇したケース。srcレイアウトを採用した上で、`uv run python src/blender-mcp-senpai/main.py`のように実行すると、`ImportError`が発生しました。

この場合、srcレイアウトを採用する前はどのパスからパッケージをインポートしていたかを特定すべきです。

ちなみに私の場合は、カレントディレクトリ経由の絶対importを、暗黙的相対importと勘違いしていたことが原因でした。

#### 非パッケージまたはフラットレイアウトからsrcレイアウトに切り替える際の注意は？

- カレントワーキングディレクトリ経由でインポートしているモジュールや静的ファイルを確認します
- `Editable install`の必要性についてドキュメントに記載します
