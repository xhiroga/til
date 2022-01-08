# 【Python C API入門】C/C++で拡張モジュール作ってPythonから呼ぶ


```shell
poetry shell

# NOTE: 動くが、正直正しいのか自信がない（.venvとそうでない場合の違いや、Poetryとvenvの関係などがよく分かっていないため）
C_INCLUDE_PATH=/usr/local/Cellar//python@3.10/3.10.1/Frameworks/Python.framework/Versions/3.10/include/python3.10/Python.h python setup.py install

pip freeze

python hello.py
```

## References and Inspirations

- [【Python C API入門】C/C\+\+で拡張モジュール作ってPythonから呼ぶ \-前編\-｜はやぶさの技術ノート](https://cpp-learning.com/python_c_api_step1/)
