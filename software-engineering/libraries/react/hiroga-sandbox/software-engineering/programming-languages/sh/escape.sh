echo `date`
# echo (date) # これは, syntax error near unexpected token `date' になる。

# シングルクォートは内容をただの文字列として扱う。ダブルクォートはエスケープを実行する。
echo '$`"\ all escaped.'