# マルチスレッド
普通はExecutorServiceインターフェースにcallableを実装したクラスを投入する。  

Javaのマルチスレッドは複数やり方がある。昔はThreadクラスやRunnableクラスを使用していた。  
CPUヘビーで再帰的な処理については、ExecutorServiceの実装の一つであるForkJoinPoolを使うとよい。　　

# 参考
http://d.hatena.ne.jp/miyakawa_taku/20171228/1514472588
