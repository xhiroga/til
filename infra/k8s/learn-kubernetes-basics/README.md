# Learn Kubernetes Basics

## Deploying an App

```shell
kubectl create deployment kubernetes-bootcamp --image=gcr.io/google-samples/kubernetes-bootcamp:v1 -v=8
kubectl delete deployment kubernetes-bootcamp -v=8
```

OR

```shell
kubectl apply -f kubernetes-bootcamp.yaml -v=8
kubectl delete -f kubernetes-bootcamp.yaml -v=8
```

## references

- [Learn Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics)
