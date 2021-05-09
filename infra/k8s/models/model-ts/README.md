# modeling kubernetes

## debug

```shell
deno run --allow-net webserver.ts

# modify ~/.kube/config by kubeconfig
kubectl config use-context model-ts
kubectl
kubectl get pods -v=8
```

## references

- [Kubernetes の Cluster 実装日記](https://zenn.dev/hiroga/scraps/ca1ec32097936b)
- [入門 Kubernetes](https://amzn.to/3uh6lKo)
- [API OVERVIEW](https://kubernetes.io/docs/concepts/overview/kubernetes-api/)
- [oakserver/oak](https://github.com/oakserver/oak)
- [vicky-gonsalves/deno_rest](https://github.com/vicky-gonsalves/deno_rest)
