# 演算子について
# $ これは引数, & これはリダイレクト先をファイル以外に扱う

undefined_command >/dev/null # これは普通に怒られる。
undefined_command >/dev/null 2>&1
undefined_command 1>/dev/null 2>/dev/null # 省略しないでこうやって書いてもいい

echo foo >/dev/null # 何も表示されない
echo foo 1>/dev/null # 何も表示されない
echo foo 2>/dev/null # これは表示される
