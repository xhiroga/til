import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse
from scrapy.utils.project import get_project_settings

from items import SteamItem


class BestSellingSpider(scrapy.Spider):
    name = 'best_selling'
    allowed_domains = ['store.steampowered.com']
    start_urls = ['https://store.steampowered.com/search/?filter=topsellers/']

    def parse(self, response: HtmlResponse):
        steam_item = SteamItem()
        games = response.xpath("//div[@id='search_resultsRows']/a")
        for game in games:
            steam_item["game_url"] = game.xpath(".//@href").get()
            steam_item["img_url"] = game.xpath(".//div[@class='col search_capsule']/img/@src").get()
            print(steam_item)


def main():
    process = CrawlerProcess(settings=get_project_settings())
    process.crawl(BestSellingSpider)
    process.start()


if __name__ == "__main__":
    main()
