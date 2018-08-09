# AWS ECS
  Docker push/pullでイメージを管理し、JSONで作成したタスクを使ってコンテナーを起動する。  

* AWS ECRはDockerのレジストリサービス(AWS版のDockerHub)
* FargateはAWSの提供するサーバレスコンテナ実行環境

# ECSの使い方
* クラスターの作成(GUI)  
  → 省略  
* タスク定義の作成(GUI)
  https://console.aws.amazon.com/ecs/  
  コンテナ追加時、リポジトリ/イメージ:タグまで入力
* Public IPの参照  
  タスク→Networkを参照　 

# ECRの使い方
* リポジトリの作成(GUI)
  リポジトリ名を入力するだけ/ イメージ1つがリポジトリ1つに相当？    
* DockerでAWSのリポジトリにログイン
  aws ecr get-login --no-include-email --region us-east-1  
* ターゲットのリポジトリ/イメージをタグ付けし、プッシュ
  docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
  docker push TARGET_IMAGE[:TAG]

# 参考
[AWS ECS](https://docs.aws.amazon.com/ja_jp/AmazonECS/latest/developerguide/docker-basics.html#use-ecr)