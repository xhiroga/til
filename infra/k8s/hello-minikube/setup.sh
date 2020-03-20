#!/bin/sh

# minikube start
# minikube dashboard &
kubectl apply -f ./deployment.yaml
kubectl apply -f ./service.yaml
kubectl get deployments
