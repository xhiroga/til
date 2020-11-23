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

```

## Reference

- https://try.jsoup.org/
- https://scrapinghub.github.io/xpath-playground/
