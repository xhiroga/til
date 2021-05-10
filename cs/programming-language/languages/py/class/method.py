print(repr(1))
print(1)


class Hero:
    name = ''


h = Hero()
h.name = 'Deku'
print(repr(h))
print(h)  # デフォルト実装では__repr__を呼び出す。


class Villan:
    name = ''

    def __repr__(self):
        return self.name


v = Villan()
v.name = 'Tomura'
print(repr(v))
print(v)
