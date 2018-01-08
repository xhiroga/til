import logging

#logging.basicConfig(filename='log.log', level=logging.INFO)

logging.warning('Watch out!')  # will print a message to the console
logging.info('I told you so')  # will not print anything
logging.debug('Hey Jude')

# 親のLoggerはDEBUGに設定し...
mylogger = logging.getLogger("mylogger")
mylogger.setLevel(logging.DEBUG)

# セットするHandlerはINFOにし、目印としてフォーマットを変えておく.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
mylogger.addHandler(ch)

mylogger.debug("Hey")
mylogger.info("Jude")

# まずLoggerインスタンスのレベルチェックがあり、次にHandlerに投げるかを決定する.
# 2018-01-07 16:41:22,132 - mylogger - INFO - Jude
# INFO:mylogger:Jude
