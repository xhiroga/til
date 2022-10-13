# envoy

```shell
brew install envoy
```

## Run Envoy with the demo configuration

```shell
envoy -c ./vendored/envoy-demo.yaml
# another terminal
curl -v localhost:10000
```

## Override the default configuration

```shell
envoy -c ./vendored/envoy-demo.yaml --config-yaml "$(cat vendored/envoy-override.yaml)"
# another terminal
curl -v localhost:9902
```

