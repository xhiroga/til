# Apache Bench

## with Authorization

```shell
export TOKEN=
ab -H "Authorization: Bearer $TOKEN" -H "content-type: application/json;charset=UTF-8" \
    -c 10 \
    -n 10000 \
    -p post.json \
    -T application/json \
    -m POST http://localhost:8080/api/v1/xxx
```
