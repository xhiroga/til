#!/bin/sh
# BシェルはどんなUNIXにも存在するため、シェルスクリプトとしてはこれを利用する

# シェル変数

# 英数字とアンダースコアが利用可能だが、1文字目が数字であってはいけない
VAR="variable"
echo "$VAR"

# 値がなにもない変数（未定義の変数とは異なる）
# 以下の2つは等価
VAR_N=
VAR_M=""
test "$VAR_N" = "$VAR_M"
echo $?

# 未定義変数
# コロンなしの変数展開で判別できる
test "${UNDEFINED+foo}"

# awk
# DSL used to data extraction and reporting.
echo "I have an apple" | awk '{ print $4 }'

# sed
# Stream EDitor, parse and transform text.
# grep(Global Regular Expression Print)の開発後にEditを目的として開発された。
:



# Reference
# [入門UNIXプログラミングシェル](https://amzn.to/33yIXvz)
# [WIKIBOOKS - Bourne Shell Scripting](https://en.wikibooks.org/wiki/Bourne_Shell_Scripting)
# [BASH man page](http://linuxjm.osdn.jp/html/GNU_bash/man1/bash.1.html)
