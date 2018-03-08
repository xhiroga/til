# Spring Batch

軽量なバッチ処理フレームワーク。停止再開、再実行・スキップなどがある。  

# 仕様

## バッチ処理完了時のステータス


## JobのStoppingとAbortingの違い
https://docs.spring.io/spring-batch/trunk/reference/htmlsingle/#stoppingAJob  
STOPPING...jobOperatorクラスが.stop()することで発生し、StepとJobの実行ステータスをBatchStatus.STOPPED  
ABOTRING...JobSriviceクラスのみ可能な操作。FAILEDと違い、再実行でも再開されない  

# チュートリアル
$ git clone https://github.com/spring-guides/gs-batch-processing.git  
$ cd gs-batch-processing/initial  
$ mvn compile  
$  # ソースコードを追加
$  ./mvnw spring-boot:run  


# 参考
Spring Batch Getting Start
https://spring.io/guides/gs/batch-processing/
