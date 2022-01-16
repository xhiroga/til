# printenv
```bash
# 環境変数を表示する
export foo=baa
printenv foo # baa

# 引数なしの場合、環境変数が全て表示される
printenv
```

# MEMO
シェル変数を環境変数にセットする場合、それは参照渡しになっている
```bash
set -a
greeting=hello
printenv greeting # hello

set +a
greeting=hi
printenc greeting # hi
```