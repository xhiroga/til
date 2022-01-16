import logging

# 1.デフォルトのログレベルはWARNING
logging.warning('Watch out!')
logging.info('I told you so')

# 2.LoggerのログレベルよりHandlerのログレベルの方が高い場合、
# 拾えなかった分は親のLoggerが処理する。
mylogger = logging.getLogger("mylogger")
mylogger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
mylogger.addHandler(ch)

mylogger.debug("Hey")
mylogger.info("Jude")

# 2018-01-07 16:41:22,132 - mylogger - INFO - Jude
# INFO:mylogger:Jude
