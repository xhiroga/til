# Python

## Language Rules / Framework Rules

### Debug

Python3.8から、f文字列で`=`指定子(`=`specifier)が使えるようになったため、積極的に用いる。[^f-string]
[^f-string]: [f-strings support = for self-documenting expressions and debugging](https://docs.python.org/3/whatsnew/3.8.html#f-strings-support-for-self-documenting-expressions-and-debugging)

```python
# Good
user = "xhiroga"
print(f"{user=}")
```

```python
# Bad
user = "xhiroga"
print(f"user={xhiroga}")
```

### Loop

$O(NM)$の処理では、よく二重ループが登場する。文字列探索におけるナイーブ法を例に、Pythonにおける個人的に好ましい書き方を考える。

```python
def partial_match(s: str, t, str) -> bool:
  i = 0
  while i < len(s):
    j = 0
    while j < len(t):
      if s[i+j] != t[j]:
        break # 文字列が一致しないことはここで明らかなのに、ループを抜けた後に一致の判定がある
      j += 1
    if j == len(t): # 一致の判定
      return True
    i += 1

  return False
```

 Pythonには、Javaにあるような`ラベル付きbreak/continue`がない。そのため、ループは内側→外側の順で抜けるしかなく、結果として内側のループを抜けたとき、ループの結果を改めて判定することになってしまう。  
 実はPythonでも`ラベル付きbreak/continue`の提案はあったようだ。[^pep3136]しかしRejectされているため、代替案として「例外処理を用いる」「内側のループを関数で括り、不一致時にはreturnする」などの方法が考えられる。けれど見づらいので、例示の通り内側のループ直後に改めて結果の判定を行うことにする。
[^pep3136]: [PEP 3136 – Labeled break and continue](https://peps.python.org/pep-3136/)

### Test

`expected`が先、`actual`が後。`unittest.TestCase`においては`assertEqual(first, second, msg=None)`のように順序を指定しない。しかし、可読性の観点から`expected`を先に書く。[^mortensen_2010]
[^mortensen_2010]: [Why are assertEquals() parameters in the order (expected, actual)?](https://stackoverflow.com/questions/2404978)

```python
assert expected == actual, print(f"{expected=}, {actual=}")
```

### Type hint

[PEP585](https://peps.python.org/pep-0585/)により、Python3.9から標準のコレクション型で型ヒントを与えられるようになったため、積極的に用いる。

## Formatting Rules

## Meta Rules

Statusがsecurity以降のバージョンを用いる。[^versions]
[^versions]: [Versions](https://devguide.python.org/versions/)

## References
