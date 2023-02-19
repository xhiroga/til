def increment():
    n = 0
    while True:
        yield n
        n += 1

"""
forループの対象として呼ぶ以外の方法だと、Generatorオブジェクトが返ってくる！

$ python 2call.py
<generator object increment at 0x10d0ab930>
<generator object increment at 0x10d0ab930>
"""
print(increment())
print(increment())


"""
next()の引数にするのが正しいけど、これは毎回新しいGeneratorオブジェクトを返してるよ？
"""
print('------------------------')
print(next(increment()))
print(next(increment()))


"""
これが正解。
<generator object increment at 0x1085aa8b8>
0
1
"""
print('------------------------')
gen = increment()
print(gen)
print(next(gen))
print(next(gen))


"""
じゃあ、Generatorオブジェクトってシングルトンになってると思う？ → なってない！
<generator object increment at 0x109ce7930>
0
<generator object increment at 0x109ce79a8>
0
"""
print('------------------------')
gen1 = increment()
print(gen1)
print(next(gen1))
gen2 = increment()
print(gen2)
print(next(gen2))
