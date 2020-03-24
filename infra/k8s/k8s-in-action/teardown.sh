#!/bin/sh

set -eux

NAMESPACE="k8s-in-action"

kubectl delete namespaces "$NAMESPACE"
