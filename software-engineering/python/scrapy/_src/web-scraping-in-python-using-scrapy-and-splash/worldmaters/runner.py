from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from worldmaters.spiders.countries import CountriesSpider


def main():
    process = CrawlerProcess(settings=get_project_settings())
    process.crawl(CountriesSpider)
    process.start()


if __name__ == "__main__":
    main()
