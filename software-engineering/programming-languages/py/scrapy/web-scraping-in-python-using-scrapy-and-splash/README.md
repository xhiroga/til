# Modern Web Scraping with Python using Scrapy Splash Selenium

https://www.udemy.com/course/web-scraping-in-python-using-scrapy-and-splash/learn/lecture/11497662#overview

## Scrapy Shell

```terminal
scrapy shell
>>> r = scrapy.Request("https://www.worldometers.info/world-population/population-by-country/")
>>> fetch(r)
>>> response.body   # fetch()のMutableなオブジェクト
>>> view(response)
```

## Run

```terminal
scrapy crawl countries -o population_datasete.json
# ('json', 'jsonlines', 'jl', 'csv', 'xml', 'marshal', 'pickle') is available

# debug
scrapy parse --spider=countries -c parse_country --meta='{"country_name":"China"}' https://www.worldometers.info/world-population/china-population/
```

## Splash

```terminal
docker-compose up -d
curl http://localhost:8050/info?wait=0.5&images=1&expand=1&timeout=90.0&url=http%3A%2F%2Fgoogle.com&lua_source=function+main%28splash%2C+args%29%0D%0A++assert%28splash%3Ago%28args.url%29%29%0D%0A++assert%28splash%3Await%280.5%29%29%0D%0A++return+%7B%0D%0A++++html+%3D+splash%3Ahtml%28%29%2C%0D%0A++++png+%3D+splash%3Apng%28%29%2C%0D%0A++++har+%3D+splash%3Ahar%28%29%2C%0D%0A++%7D%0D%0Aend

# down
docker-compose down
```

## Selenium

```terminal
brew install chromedriver
```

## Reference

- https://try.jsoup.org/
- https://scrapinghub.github.io/xpath-playground/
