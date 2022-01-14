# AWS Batch

# Usage
1. コンテナをECR(またはDockerHub)に登録する。
2. コンピューティング環境を作成する。
3. ジョブキューを作成する。
4. ジョブ定義を作成する。
 → ここでロールをジョブロール候補に表示させるためには、ECSのTask実行に関するポリシーがアタッチされている必要がある。


# TIPS
* Alpineのashが起動時にprofileを読みこむのは、`-l`オプションでlogin shellとして開いた場合のみ。
* Minimum vCPUを0にしておくと、インスタンスが起動後1時間しない程度でterminateしてくれる。
* ジョブ実行のために必要な権限は、"コンピューティング環境"の"インスタンスロール"に付与する...のではなく、"ジョブ定義"の"ジョブロール"に付与する。
* "コンピューティング環境"ごとに自動的にECSのクラスターが作成される。
* コンテナイメージの指定は、ECRのものを利用する場合長いURIを指定する。もしコンテナが見つからないと10分くらい探してからエラーになるので疲れる。
* 起動時のコマンドと環境変数はジョブ定義で指定できるので、コンテナ作成時にそこまでカリカリしなくていい。

# Reference
https://dev.classmethod.jp/cloud/aws-batch-5min-over/
[AWS Batch ユーザーガイド](https://docs.aws.amazon.com/ja_jp/batch/latest/userguide/Batch_GetStarted.html)