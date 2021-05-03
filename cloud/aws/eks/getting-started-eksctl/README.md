# Getting started with Amazon EKS – eksctl

```shell
eksctl create cluster \
--name hiroga-cluster \
--region ap-northeast-1 \
--fargate

kubectl config use-context Administrator@hiroga-cluster.ap-northeast-1.eksctl.io

```

## references

- [Getting started with Amazon EKS – eksctl](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html)
