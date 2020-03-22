#!/bin/sh

# kubectl apply -f simple-pod.yaml
kubectl apply -f simple-replicaset.yaml
echo 'If you want to enter the container, run "kubectl exec -it simple-echo sh -c nginx"'

printf "\nNode (k8sクラスタの管理化にあるDockerホスト)\n\n"
kubectl get nodes

printf "\nNamespaces (仮想的なクラスタ)\n\n"
kubectl get namespace
