"""
@dataclassモジュールはデコレータとして使うことを想定されている。__init__や__repr__を書かなくてよくなる。
メンバー変数は型アノテーションを用いて提供される。
"""
import dataclasses


@dataclasses.dataclass
class Villan:
    name: str
    age: int


v = Villan('Tomura', None)  # メンバーの初期値を引数にとるコンストラクタを勝手に設定してくれる。
print(v)  # Villan(name='Tomura', age=None) とてもわかりやすい。

# dataclassesデコレータが提供する便利なユーティリティが利用できるのもメリット。
print(dataclasses.fields(v))


class Hero:
    # もちろん、型アノテーションによる宣言はdataclassデコレータとは関係なく使える。
    name: str
    age: int


h = Hero()
h.name = 'Deku'
h.age = '15'
print(h)
