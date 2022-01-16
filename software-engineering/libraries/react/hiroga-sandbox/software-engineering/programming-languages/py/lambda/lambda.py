"""
do not assign a lambda expression, use a def

JavaScriptとの違い: 
* 変数に代入してはいけない
* 複数行になってはいけない
"""
func = lambda x, y: x + y
print(func(1,2))
