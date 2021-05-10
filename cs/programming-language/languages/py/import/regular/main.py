from string import instruments
instruments.violin()

# 以下の処理はエラーとなる。stringが同じフォルダ内のregular packageで上書きされているため、標準ライブラリのstringパッケージを探しに行けない。
# from string import digits
# print(digit)
