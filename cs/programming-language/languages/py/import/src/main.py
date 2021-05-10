# importについて
"""
Pythonがimportをするとき、そのモジュールを探す先（＝sys.path）は、

1. 実行中のファイルと同じフォルダ。未指定ならカレントフォルダ。
2. PYTHONPATHに指定されたフォルダ
3. インストールごとのデフォルト
"""

from awesome import awesome
awesome()

# python src/import.pyとして起動する限り、これはimportできない。
# from current import currnet
# currnet()

import sys

print(sys.path)
"""
4. sys.pathに登録されているフォルダについて。ここからライブラリをimportしている。
['/Users/hiroaki/Dev/til/lang/py/import/src', '/Users/hiroaki/.pyenv/versions/3.7.0/lib/python37.zip', '/Users/hiroaki/.pyenv/versions/3.7.0/lib/python3.7', '/Users/hiroaki/.pyenv/versions/3.7.0/lib/python3.7/lib-dynload', '/Users/hiroaki/.pyenv/versions/3.7.0/lib/python3.7/site-packages']
"""


# packageについて
from nest.directory.file import nested_method
nested_method()
