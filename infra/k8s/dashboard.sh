#!/bin/sh

kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta8/aio/deploy/recommended.yaml
kubectl proxy &
kubectl -n kubernetes-dashboard describe secrets default | grep 'token:' | awk '{ print $2 }'
open http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
