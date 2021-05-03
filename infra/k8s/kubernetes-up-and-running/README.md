# Kubernetes: Up and Running

## 5. Pod

```shell
kubectl apply -f kuard-pod.yaml -v=8
kubectl get pods -o=wide -v=6
kubectl delete -f kuard-pod.yaml -v=8

kubectl port-forward kuard 8080:8080 -v=9 # Level 8 truncates log
```

YAML の内容がそのまま POST される。
YAML のプロパティと API にリクエストする際のパスは一部情報が重複するが、コードとしての管理しやすさと RESTful な I/F を両立するためには必要なのだろう。

### 5.6. ヘルスチェック

```shell
kubectl apply -f kuard-pod-health.yaml -v=8
kubectl port-forward kuard 8080:8080
open http://localhost:8080/-/liveness
# k3sで実行した場合、10秒ごとにヘルスチェックのリクエストが実行されるようだ。

kubectl delete -f kuard-pod-health.yaml
```

プロセスヘルスチェックの場合、プロセスがデッドロックを起こしている場合にも問題ないと判断されてしまう。
これを回避するために、アプリケーション固有のロジックによるヘルスチェックを設定できる。Liveness(=活性）ヘルスチェック

### 5.8. ボリューム

```shell
kubectl apply -f kuard-pod-vol.yaml -v=8
kubectl delete -f kuard-pod-vol.yaml
```

## references

- [入門 Kubernetes](https://amzn.to/3aTfAZp)
