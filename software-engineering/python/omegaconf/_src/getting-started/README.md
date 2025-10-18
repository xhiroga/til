# OmegaConf

YAMLベースの設定パーサー。YAMLとCLIなどの複数のソースから設定を合成できる。

Hydraの内部で利用されているため、Meta系の機械学習パッケージで必要になることがある（fairseqなど）

## How to run


## トラブルシューティング

### omegaconf.errors.ConfigKeyError

structured configに存在しないキーの合成を試みた際のエラー。

```py
base_conf: DictConfig = OmegaConf.structured(MyConfig)

print(OmegaConf.to_yaml(base_conf))

log_cfg = OmegaConf.create({"log": {"file": "log.txt"}})
merged_conf = OmegaConf.merge(base_conf, log_cfg)
print(OmegaConf.to_yaml(merged_conf))
```

```console
omegaconf.errors.ConfigKeyError: Key 'log' not in 'MyConfig'
    full_key: log
    object_type=MyConfig
```
