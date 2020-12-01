from shutil import which

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse
from scrapy.selector import Selector
from scrapy.utils.project import get_project_settings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class CoinSeleniumSpider(scrapy.Spider):
    name = 'coin_selenium'
    allowed_domains = ['www.livecoin.net/en']
    start_urls = [
        "https://www.livecoin.net/en"
    ]

    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")

        chrome_path = which("chromedriver")

        driver = webdriver.Chrome(executable_path=chrome_path, options=chrome_options)
        driver.set_window_size(1920, 1080)  # 画面サイズを大きくしないとコンテンツがロードされない場合がある
        driver.get("https://www.livecoin.net/en")

        # tabs = driver.find_element_by_class_name("filterPanelItem___2z5Gb")  # works like first()
        tabs = driver.find_elements_by_class_name("filterPanelItem___2z5Gb")
        litecoin_tab = tabs[5]
        litecoin_tab.click()

        self.html = driver.page_source
        driver.close()

    def parse(self, response: HtmlResponse):
        resp = Selector(text=self.html)
        for currency in resp.xpath("//div[contains(@class, 'ReactVirtualized__Table__row tableRow___3EtiS ')]"):
            yield {
                'currency pair': currency.xpath(".//div[1]/div/text()").get(),
                'volume(24h)': currency.xpath(".//div[2]/span/text()").get()
            }


def main():
    process = CrawlerProcess(settings=get_project_settings())
    process.crawl(CoinSeleniumSpider)
    process.start()


if __name__ == "__main__":
    main()
