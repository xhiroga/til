# Kubernetes: Up and Running

## 5. Pod

```shell
kubectl apply -f pod/kuard-pod.yaml -v=8
kubectl get pods -o=wide -v=6
kubectl delete -f pod/kuard-pod.yaml -v=8

kubectl port-forward kuard 8080:8080 -v=9 # Level 8 truncates log
```

YAML の内容がそのまま POST される。
YAML のプロパティと API にリクエストする際のパスは一部情報が重複するが、コードとしての管理しやすさと RESTful な I/F を両立するためには必要なのだろう。

### 5.6. ヘルスチェック

```shell
kubectl apply -f pod/kuard-pod-health.yaml -v=8
kubectl port-forward kuard 8080:8080
open http://localhost:8080/-/liveness
# k3sで実行した場合、10秒ごとにヘルスチェックのリクエストが実行されるようだ。

kubectl delete -f pod/kuard-pod-health.yaml
```

プロセスヘルスチェックの場合、プロセスがデッドロックを起こしている場合にも問題ないと判断されてしまう。
これを回避するために、アプリケーション固有のロジックによるヘルスチェックを設定できる。Liveness(=活性）ヘルスチェック

### 5.8. ボリューム

```shell
kubectl apply -f pod/kuard-pod-vol.yaml -v=8
kubectl delete -f pod/kuard-pod-vol.yaml
```

## 6. Label と Annotation

```shell
kubectl run alpaca-prod --image=gcr.io/kuar-demo/kuard-amd64:1 --labels="ver=1,app=alpaca,env=prod" -v=8
kubectl run alpaca-test --image=gcr.io/kuar-demo/kuard-amd64:2 --labels="ver=2,app=alpaca,env=test" -v=8
kubectl run bandicoot-prod --image=gcr.io/kuar-demo/kuard-amd64:2 --labels="ver=2,app=bandicoot,env=prod" -v=8
kubectl run bandicoot-staging --image=gcr.io/kuar-demo/kuard-amd64:2 --labels="ver=2,app=bandicoot,env=staging" -v=8
kubectl get deployments --show-labels -v=9
```

## 8. ReplicaSet

```shell
kubectl apply -f replica-set/kuard-rs.yaml -v=8
kubectl scale replicasets kuard --replicas=4 -v=8

kubectl delete -f replica-set/kuard-rs.yaml -v=8
```

## 12. Deployment

```shell
kubectl apply -f deployment/nginx-deployment.yaml -v=9
kubectl delete -f deployment/nginx-deployment.yaml -v=8
```

## references

- [入門 Kubernetes](https://amzn.to/3aTfAZp)
