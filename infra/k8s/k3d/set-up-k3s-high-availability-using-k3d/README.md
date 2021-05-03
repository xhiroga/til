# Set up K3s in High Availability using k3d

## set up

```shell
k3d cluster create # create one node cluster

kubectl config get-contexts

# before describe by kubectl, switch context to local k3s
kubectl config use-context k3d-k3s-default

kubectl get all --all-namespaces -v=6

k3d cluster delete
```

### High Availability Cluster

```shell
k3d cluster create --servers 3 --image rancher/k3s:v1.19.3-k3s2
kubectl get nodes --output wide -v=6
```

```shell
helm repo add stable https://charts.helm.sh/stable
```

## references

- [Set up K3s in High Availability using k3d](https://rancher.com/blog/2020/set-up-k3s-high-availability-using-k3d)
- [Local Kubernetes Development](https://medium.com/@lukejpreston/local-kubernetes-development-a14ea8be54d6)
