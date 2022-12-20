# Finch

```shell
finch vm start
# M1 Macの場合、 qemu-system-aarch64 が起動する。メモリ使用量2.9GB。
finch run --rm public.ecr.aws/finch/hello-finch

finch run --name nginx -p 8080:80 -v $(pwd)/contents:/usr/share/nginx/html:ro nginx:latest
# メモリ使用量 2.29GB → 2.53GB に増加。

finch stop
# qemu-system-aarch64 が終了し、Activity Monitorから見つからなくなる。
```

## References

 - [AWSが公開したFinchでコンテナ実行/イメージビルドをする | 豆蔵デベロッパーサイト](https://developer.mamezou-tech.com/blogs/2022/12/05/finch-intro/)
