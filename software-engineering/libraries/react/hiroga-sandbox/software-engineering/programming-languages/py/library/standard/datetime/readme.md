# Pythonと日時
日時をオブジェクトとして持つなら、個人的には標準時のオブジェクト+TimeZoneが妥当と考える。
Pythonはそうではない。明示しなければローカル日時。TimeZoneを持たせるかは任意。  
→ つまり、同じ瞬間を表すことのできるDateTimeオブジェクトが複数ある。  


## DateTimeの生成
→ dt.now(pytz.utc)でOK. 安全のためTimeZone必須とする.  
```
from datetime import datetime as dt
from datetime import timedelta
from datetime import
dt.today() # ローカルな現在日時
dt.now(timezone(timedelta(hours=9))) # ローカルな現在日時（タイムゾーンつき）
# dt.now()に渡すのはtzinfoのサブクラスならなんでもいいので、サードパーティのpytzのオブジェクトでもOK
dt.now(pytz.timezone('Asia/Tokyo'))

dt.now(pytz.utc) # グローバルな現在日時のツウなやり方(タイムゾーンつきなので安心)
```


## タイムゾーンありなしの見た目
→ いい感じにタイムゾーンをつけてくれる
```
>>> print(dt.today())
2018-01-26 21:30:11.190465
>>> print(dt.now(timezone(timedelta(hours=9))))
2018-01-26 21:30:29.619718+09:00
```


## nativeからawareへの変更
→ pytzがtimezone.localize()を用意してくれている。  
```
>>> pytz.utc.localize(dt.utcnow())
datetime.datetime(2018, 1, 26, 23, 57, 18, 731087, tzinfo=<UTC>)
```


## 違う現地時間への変換
→ datetime.astimezone()が用意されている。  
```
>>> now = dt.now(pytz.utc)
datetime.datetime(2018, 1, 26, 13, 8, 53, 226796, tzinfo=<UTC>)
>>> jst = now.astimezone(pytz.timezone('Asia/Tokyo'))
>>> jst
datetime.datetime(2018, 1, 26, 22, 8, 53, 226796, tzinfo=<DstTzInfo 'Asia/Tokyo' JST+9:00:00 STD>)
>>> jst.astimezone(pytz.timezone('Asia/Tokyo'))
datetime.datetime(2018, 1, 26, 13, 8, 53, 226796, tzinfo=<UTC>)
```


## タイムゾーンの異なるdatetimeの比較
→ 正しく比較される。  
ex1) 日本の深夜0時とイギリスの前日22時
```
>>> j = dt(2018,1,17,0,0,0,tzinfo=pytz.timezone('Asia/Tokyo'))
>>> e = dt(2018,1,16,22,0,0,tzinfo=pytz.utc)
>>> if e > j: print(True)
True
```


## その他
https://stackoverflow.com/questions/12255932/what-does-the-p-in-strptime-stand-for
