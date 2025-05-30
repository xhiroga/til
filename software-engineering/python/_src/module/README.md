# Module

Pythonにおいて、`.py` ファイルをモジュール、複数のモジュールをまとめたものをパッケージと呼ぶ。

## import

`sys.path`以下のパスに含まれるモジュール・パッケージをimportできる。

```console
$ python -c "import sys; print(sys.path)"
# OR
$ uv run python -c "import sys; print(sys.path)"

['', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python313.zip', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13', '/home/hiroga/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/lib/python3.13/lib-dynload', '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/.venv/lib/python3.13/site-packages', '/home/hiroga/Documents/GitHub/til/software-engineering/python/_src/module/src']
```
