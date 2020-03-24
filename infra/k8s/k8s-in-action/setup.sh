#!/bin/sh

set -eux

printf "\nNode (k8sクラスタの管理化にあるDockerホスト)\n\n"
kubectl get nodes

printf "\nNamespaces (仮想的なクラスタ)\n\n"
NAMESPACE="k8s-in-action"
if [ ! "$(kubectl get namespace | grep $NAMESPACE)" ]
then
    kubectl create -f ./namespace.yaml
fi
kubectl get namespace

printf "\nRun Deployment\n\n"
# kubectl apply -f simple-pod.yaml                          # Replicas of pods are defined by ReplicaSet
# kubectl apply -f simple-replicaset.yaml                   # Generations of ReplicaSets are defined by deployment
# kubectl apply -f simple-deployment.yaml --record=true     # Pod数の管理や新しいバージョンのPodへの交代を世代管理するためにDeploymentを用いる
kubectl apply -n "$NAMESPACE" -f simple-replicaset-with-label.yaml --record=true
printf "If you want to enter the container, run \"kubectl exec -it simple-echo sh -c nginx\"\n"
kubectl get pod,replicaset,deployment -n "$NAMESPACE" --selector app=echo

# kubectl rollout history deployment echo
# ReplicaSetの置換が発生した場合のみRevisionが+1される

kubectl apply -n "$NAMESPACE" -f simple-service.yaml --record=true      # Service（サービスディスカバリなど）の作成
kubectl get svc -n "$NAMESPACE" -l app=echo
printf "If you want to request to service, run \"kubectl run -i --rm --tty debug --image=gihyodocker/fundamental:0.1.0 --restart=Never -- bash -il\"\n"
