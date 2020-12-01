from typing import Optional

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse
from scrapy.utils.project import get_project_settings
from scrapy.utils.response import open_in_browser
from scrapy_selenium import SeleniumRequest


def remove_characters(value: Optional[str]):
    if value:
        return value.strip('\xa0')
    else:
        return None


class ComputerDealsSpider(scrapy.Spider):
    name = 'computer-deals'

    def start_requests(self):
        yield SeleniumRequest(
            url="https://slickdeals.net/computer-deals/",
            wait_time=3,
            callback=self.parse
        )

    def parse(self, response: HtmlResponse):
        # open_in_browser(response)
        products = response.xpath("//li[@class='fpGridBox grid altDeal hasPrice']")
        for product in products:
            yield {
                "name": product.xpath(".//a[contains(@class, 'itemTitle')]/text()").get(),
                "link": product.xpath(".//a[contains(@class, 'itemTitle')]/@href").get(),
                "store_name": remove_characters(product.xpath(".//a[contains(@class, 'itemStore')]/text()").get()),
                "price": product.xpath("normalize-space(.//div[contains(@class, 'itemPrice')]/text())").get(),
            }

        next_page = response.xpath("//a[@data-role='next-page']/@href").get()
        if next_page:
            absolute_url = f"https://slickdeals.net/{next_page}"
            yield SeleniumRequest(
                url=absolute_url,
                wait_time=3,
                callback=self.parse
            )


def main():
    process = CrawlerProcess(settings=get_project_settings())
    process.crawl(ComputerDealsSpider)
    process.start()


if __name__ == "__main__":
    main()
