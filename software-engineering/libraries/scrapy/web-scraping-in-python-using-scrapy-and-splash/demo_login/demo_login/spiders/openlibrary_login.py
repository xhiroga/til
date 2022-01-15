import os

import scrapy
from scrapy import FormRequest
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


class OpenlibraryLoginSpider(scrapy.Spider):
    name = 'openlibrary_login'
    allowed_domains = ['openlibrary.org/account/login']
    start_urls = ['https://openlibrary.org/account/login']

    def parse(self, response):
        yield FormRequest.from_response(
            response,
            formid="register",
            formdata={
                "username": os.environ.get("OPENLIB_USERNAME"),
                "password": os.environ.get("OPENLIB_PASSWORD"),
                "redirect": "/",
                "debug_token": "",
                "login": "Log In"
            },
            callback=self.after_login
        )

    def after_login(self, response):
        print("logged in...")


def main():
    process = CrawlerProcess(settings=get_project_settings())
    process.crawl(OpenlibraryLoginSpider)
    process.start()


if __name__ == "__main__":
    main()
