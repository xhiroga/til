#!/bin/sh

set -eux

printf "\nNode (k8sクラスタの管理化にあるDockerホスト)\n\n"
kubectl get nodes

printf "\nNamespaces (仮想的なクラスタ)\n\n"
kubectl get namespace

printf "\nRun Deployment\n\n"
# kubectl apply -f simple-pod.yaml                          # Replicas of pods are defined by ReplicaSet
# kubectl apply -f simple-replicaset.yaml                   # Generations of ReplicaSets are defined by deployment
# kubectl apply -f simple-deployment.yaml --record=true     # Pod数の管理や新しいバージョンのPodへの交代を世代管理するためにDeploymentを用いる
kubectl apply -f simple-replicaset-with-label.yaml
printf "If you want to enter the container, run \"kubectl exec -it simple-echo sh -c nginx\"\n"
kubectl get pod,replicaset,deployment --selector app=echo

kubectl rollout history deployment echo
# ReplicaSetの置換が発生した場合のみRevisionが+1される

kubectl apply -f simple-service.yaml                        # Service（サービスディスカバリなど）の作成
kubectl get svc echo
printf "If you want to request to service, run \"kubectl run -i --rm --tty debug --image=gihyodocker/fundamental:0.1.0 --restart=Never -- sbash -il\"\n"
