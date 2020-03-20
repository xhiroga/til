# Kubernetes(クーベネティス)

Scalable な Microservices のためのインフラストラクチャー  
アプリケーションをモジュールに分割し(portability)、デプロイを簡単にし(deployability)、一部分だけをスケールできるようにする(scalability)。

SaaS アプリケーション開発には 12 の原則からなる方法論がある。  
https://12factor.net/ja/

JWT(ジョート)... JSON Web Tokens  
2 者間で安全に情報をやり取りするための規格。base64Encode された Header+Payload と、それらに秘密鍵を添付して作成したハッシュ(=サイン)からなる。  
jwt.io で簡単に Encode/Decode 可能。  
https://jwt.io/

## GCP

以下の二つを有効にする必要がある。

- Google Compute Engine API
- Google Kubernetes Engine API  
  かつ、クラスタを作成する必要あり。

```console:
gcloud container clusters create k0 --zone “us-central1-a”
```

# Pods

1 つ以上のコンテナとボリュームからなる、アプリケーションの単位。  
ex) nginx コンテナと app コンテナ

```console:
cat ngingx-pod.yaml # yamlファイルで構成を管理
kubectl create -f pods/nginx-pod.yaml # Podを作成(この時点で起動)
kubectl get pods
kubectl port-forward nginx-pod 10080:80
kubectl logs -f nginx-pod # ログをウォッチ
kubectl exec nginx-pod --stdin --tty -c nginx-pod /bin/sh # シェルを実行
```

# 参考資料

- Udacity のコース  
  https://classroom.udacity.com/courses/ud615

- Udacity で紹介している Resource 一覧
  https://classroom.udacity.com/courses/ud615/lessons/7826112332/concepts/80841806450923

- k8s のサンプル  
  https://github.com/udacity/ud615

- [Docker/Kubernetes 実践コンテナ開発入門](https://amzn.to/2QvmkTm)
