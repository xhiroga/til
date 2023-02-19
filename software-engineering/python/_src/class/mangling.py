class AHero:
    """A Class"""
    name = 'Batman'
    __hidden_name = 'Bruce'


c = AHero()
print(c.name)
# print(c.__hidden_name) # Error
print(c._AHero__hidden_name)  # 警告が出る。mypyは名前マングリングをサポートしないらしい？

# あるメソッドが外側からも内側からも利用されている時で、サブクラスが外側からの挙動だけを変えたい時、
# メインクラスは外側からアクセスされるメソッドのコピーを内側にコピーしておくことで、サブクラスによるoverrideが可能になる。
