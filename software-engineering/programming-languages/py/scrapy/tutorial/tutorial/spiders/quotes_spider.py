import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    # scrapy crawl quotes does
    # 1. Load spider by sp(ider)name
    # 2. Invoke start_requests
    # https://github.com/scrapy/scrapy/blob/5e9cc3298be4ca1146d68098899365b334706db8/scrapy/cmdline.py#L109
    def start_requests(self):
        urls = [
            "http://quotes.toscrape.com/page/1/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Parser is implemented in different repository from scrapy itself.
        # https://github.com/scrapy/parsel
        for quote in response.css("div.quote"):
            yield {
                "text": quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
                "tags": quote.css("div.tags a.tag::text").get(),
            }
        next_page = response.css("li.next a::attr(href)").get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
