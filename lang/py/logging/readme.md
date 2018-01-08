# チートシート
こんな感じで設定すればOK.

```
logger = logging.getLogger(__name__)
logger.setLever(logger.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.setHandler(ch)

```

基本方針: 
* Loggerインスタンスで出力する
* #debug()で出力し、productコードではsetLevel(logger.WARNING)する
* Handlerは面倒でもつける、なぜならfomatterはHandlerにしかセットできない
* 時間はデバッグの重要手がかりなのでここからコピペして使う.


# メモ
ルートロガーまたはLoggingオブジェクトのメソッドを呼び出す。

デフォルトのレベルはWARNING.
Logger.basicConfig(filename="",level="") で出力ファイル指定.
Logger.setLevel(logging.INFO) でレベル設定

logger#getLogger()に同じ名前を渡せば、同じLoggingインスタンスが返る.
(関数内でLoggerインスタンス生成とログ出力を定義し、外側で同じ名前で呼び出したLoggerインスタンスのレベルを操作したところ、関数内の動作に影響した.)


# 参考資料
https://docs.python.jp/3/howto/logging.html
