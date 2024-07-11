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

### Test

`expected`が先、`actual`が後。`unittest.TestCase`においては`assertEqual(first, second, msg=None)`のように順序を指定しない。しかし、可読性の観点から`expected`を先に書く。[^mortensen_2010]
[^mortensen_2010]: [Why are assertEquals() parameters in the order (expected, actual)?](https://stackoverflow.com/questions/2404978)

```py
assert expected == actual, print(f"{expected=}, {actual=}")
```

### Type hint

[PEP585](https://peps.python.org/pep-0585/)により、Python3.9から標準のコレクション型で型ヒントを与えられるようになったため、積極的に用いる。

## Formatting Rules

## Meta Rules

Statusがsecurity以降のバージョンを用いる。[^versions]
[^versions]: [Versions](https://devguide.python.org/versions/)

## References
