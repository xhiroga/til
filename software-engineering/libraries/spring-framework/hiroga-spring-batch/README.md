# Spring Batch

軽量なバッチ処理フレームワーク。停止再開、再実行・スキップなどがある。  

# 仕様

## バッチ処理のステータス
通常のBatchStatusの遷移  
Starting → Started → Completed/ Failed/ Unknown  

手動の介入があった場合  
Starting/Started のStep/Job に対して、Stop命令を送るとStopping/Stopped になる。  
StoppingまたはStoppedのStep/Jobに対して、Abandon命令を送るとAbandoned になる。  

## Stopの仕様
おそらく次のような仕様になっている。  
1. JobOperator#stopでexecutionIdを指定し、ジョブのBatchStatusをStoppingに変更する。  
2. JobExecutionインスタンスはチャンク境界(StepとStepの間)ごとに自身のBatchStatusを監視しており、それがStoppingなら停止処理に入る。  
3. 停止処理では、まずBATCH_JOB_EXECUTIONとBATCH_STEP_EXECUTIONのSTATUSをSTOPPINGに変更する。  
4. 次に現在のチャンク境界でSTEPの処理を停止し、BATCH_STEP_EXECUTIONのSTATUSをSTOPPEDに変更する。  
5. （ここは自信ない）それ以降のSTEPの処理も全て停止し、BATCH_STEP_EXECUTIONのSTATUSをSTOPPEDに変更する。  
6. ジョブは全てのSTEPがSTOPPEDになっていたら、自身もSTOPPEDにする。  

# チュートリアル
$ git clone https://github.com/spring-guides/gs-batch-processing.git  
$ cd gs-batch-processing/initial  
$ mvn compile  
$  # ソースコードを追加
$  ./mvnw spring-boot:run  


# 参考
Spring Batch Getting Start  
https://spring.io/guides/gs/batch-processing/  

Spring Batch: BatchStatus state transitions  
https://blog.codecentric.de/en/2014/04/spring-batch-batchstatus-state-transitions/  
