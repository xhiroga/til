# PostgresSQL

```shell
psql URI # 接続  
# URIは環境変数をそのまま指定すると楽チン($ psql $DATA_BASE)  

postgres=> \d # lsに相当  
# ただし、set pathしないとpublic以外は検索対象にならない  

# バックアップを取得する.   
pg_dump $DATA_BASE >> dump.sql  
```

## References
https://www.postgresql.org/docs/10/static/app-psql.html
