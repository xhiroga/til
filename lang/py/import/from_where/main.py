import sys
print(__name__)
print(sys.path)

from awesome import awesome
awesome()

# 上記の処理を、moudle内でも同様に行う。
from pkg import module
module.execute()

# 全く同じ結果が返ってくる。
