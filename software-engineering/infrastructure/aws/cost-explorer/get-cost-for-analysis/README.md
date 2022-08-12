# Get cost for analysis

コスト分析のために、過去半年のAWSコストを取得する

```shell
./get-cost > out/$(gdate -d '-1 month' '+%Y-%m').csv
```
