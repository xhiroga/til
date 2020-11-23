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

## Reference

- https://try.jsoup.org/
- https://scrapinghub.github.io/xpath-playground/
