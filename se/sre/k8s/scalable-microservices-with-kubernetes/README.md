# Kubernetes(クーベネティス)

Scalable な Microservices のためのインフラストラクチャー  
アプリケーションをモジュールに分割し(portability)、デプロイを簡単にし(deployability)、一部分だけをスケールできるようにする(scalability)。

SaaS アプリケーション開発には 12 の原則からなる方法論がある。  
https://12factor.net/ja/

JWT(ジョート)... JSON Web Tokens  
2 者間で安全に情報をやり取りするための規格。base64Encode された Header+Payload と、それらに秘密鍵を添付して作成したハッシュ(=サイン)からなる。  
jwt.io で簡単に Encode/Decode 可能。  
https://jwt.io/

## gcp

Before starting tutorial, make apis enable and create cluster.

```shell
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud container clusters create k0 --zone “us-central1-a”
```

## pods

1 つ以上のコンテナとボリュームからなる、アプリケーションの単位。  
ex) nginx コンテナと app コンテナ

```shell
cat pods/nginx-pod.yaml # yamlファイルで構成を管理
kubectl create -f pods/nginx-pod.yaml -v=8 # Podを作成(この時点で起動)
kubectl get pods -v=6
kubectl port-forward nginx-pod 10080:80
kubectl logs -f nginx-pod # ログをウォッチ
kubectl exec nginx-pod --stdin --tty -c nginx-pod /bin/sh # シェルを実行

kubectl delete pod nginx-pod -v=8
```

## references

- [入門 Kubernetes](https://amzn.to/3nG7ybG)
- [Scalable Microservices with Kubernetes - Udacity](https://classroom.udacity.com/courses/ud615)
- [Resources - Udacity](https://classroom.udacity.com/courses/ud615/lessons/7826112332/concepts/80841806450923)
- [UD615: Scalable Microservices with Kubernetes](https://github.com/udacity/ud615)
