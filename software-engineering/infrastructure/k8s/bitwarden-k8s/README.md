# bitwarden-k8s

## How to run

```shell
kubectl config use-context k3d-k3s-default
helm repo add bitwarden https://cdwv.github.io/bitwarden-k8s/
helm install bitwarden bitwarden/bitwarden-k8s
```
