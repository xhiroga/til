# Datadog LightHouse Integration with Docker

**Work in progress.**

```shell
make run

# Inside Docker
sudo lighthouse https://www.datadoghq.com --output json --quiet --chrome-flags='--headless --no-sandbox --disable-gpu --disable-dev-shm-usage'

# It fails with `Unable to connect to Chrome`
sudo -u dd-agent lighthouse https://www.datadoghq.com --output json --quiet --chrome-flags='--headless --no-sandbox --disable-gpu --disable-dev-shm-usage'
```

## References and Inspirations

- [Lighthouse](https://docs.datadoghq.com/integrations/lighthouse/)
- [Use Community Integrations](https://docs.datadoghq.com/agent/guide/use-community-integrations/?tab=docker)
- [Tags · DataDog/integrations\-extras](https://github.com/DataDog/integrations-extras/tags)
- [Docker Agent](https://docs.datadoghq.com/agent/docker/?tab=standard)
- [Docker で Puppeteer を動かす \- Qiita](https://qiita.com/athagi/items/305f55bc140683d3dca7)
