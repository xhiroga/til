# podman


## Run

```shell
podman run -dt -p 8080:80/tcp docker.io/library/httpd
podman ps
curl http://localhost:8080
```

## Build

```shell
podman build -f Dockerfile -t fedora-httpd .
```

## References
- [Getting Started with Podman](https://podman.io/getting-started/)
