import scrapy
from scrapy import Selector

from scrapy_selenium import SeleniumRequest
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.keys import Keys


class ExampleSpider(scrapy.Spider):
    name = 'example'

    def start_requests(self):
        yield SeleniumRequest(
            url="https://duckduckgo.com",
            wait_time=3,
            screenshot=True,
            callback=self.parse
        )

    def parse(self, response):
        # img = response.meta["screenshot"]
        #
        # with open("screenshot.png", "wb") as f:
        #     f.write(img)

        driver: WebDriver = response.meta["driver"]
        search_input = driver.find_element_by_xpath("//input[@id='search_form_input_homepage']")
        search_input.send_keys("Hello World")

        search_input.send_keys(Keys.Enter)      # Initial response is modified

        html = driver.page_source
        response_obj = Selector(text=html)

        # driver.save_screenshot("after_filling_input.png")

        links = response_obj.xpath("//span[@class='result__url__full']")
        for link in links:
            yield {
                "URL": link.xpath(".//@href")
            }
