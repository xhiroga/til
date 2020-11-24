import scrapy
from scrapy import FormRequest


# from scrapy.utils.response import open_in_browser


class QuotesLoginSpider(scrapy.Spider):
    name = 'quotes_login'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['https://quotes.toscrape.com/login']

    def parse(self, response):
        csrf_token = response.xpath("//input[@name='csrf_token']/@value").get()
        # open_in_browser(response)
        yield FormRequest.from_response(
            response,
            formxpath="//form",
            formdata={
                "csrf_roken": csrf_token,
                "username": "admin",
                "password": "admin"
            },
            callback=self.after_login
        )

    def after_login(self, response):
        if response.xpath("//a[@href='/logout']/text()"):
            print("logged in")
