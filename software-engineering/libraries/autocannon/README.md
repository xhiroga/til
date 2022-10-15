# AutoCannon

## with Authorization

```shell
export TOKEN=
autocannon -H "Authorization: Bearer $TOKEN" -H "content-type: application/json;charset=UTF-8" \
    -c 10 \
    -d 5 \  # OR -a 10000   # 秒数 OR リクエスト数を指定
    -p 1 \
    --renderStatusCodes \
    -m POST -b '{}' 
```
