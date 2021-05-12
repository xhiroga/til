# 正規表現
Pythonの正規表現を使った文字列比較はreモジュールのモジュール関数(≒クラスメソッド)か、正規表現オブジェクトのメソッドで行う。  
文字列の検証やトリミングには`search()`や`match()`を用いる。`match()`は、対象の文字列が複数行であろうが先頭からしかマッチしない。    

# Usage
## モジュールコンテンツ
```
python
>>> import re
>>> m = re.match(r"bitcoin:(12345)", "bitcoin:12345")
>>> m.group(0) # 'bitcoin:12345'
>>> m.group(1)# '12345'
```

## 正規表現オブジェクト
```
python
>>> import re
>>> pat = re.compile(r"bitcoin:(12345)")
>>> m = pat.match("bitcoin:12345")
>>> m.group(0) # 'bitcoin:12345'
>>> m.group(1)# '12345'
```
