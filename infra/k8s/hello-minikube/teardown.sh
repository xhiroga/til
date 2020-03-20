#!/bin/sh

kubectl delete -n default deployment hello-node
minikube stop
