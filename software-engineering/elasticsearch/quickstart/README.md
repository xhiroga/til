# Quick Start

## Run

```shell
docker compose up
open http://localhost:5601/app/dev_tools#/console
curl -X GET http://localhost:9200/
```

## Delete

```shell
docker compose down
docker network rm elastic
```


## References

[Elastic Docs ›Elasticsearch Guide [7.17] - Quick start](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/getting-started.html)
