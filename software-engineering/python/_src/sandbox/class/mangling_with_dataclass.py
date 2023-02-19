import dataclasses


@dataclasses.dataclass
class Villan:
    __name: str
    __age: int


v = Villan('Kurohige', None)

# print(v.__name)
print(v._Villan__name)
