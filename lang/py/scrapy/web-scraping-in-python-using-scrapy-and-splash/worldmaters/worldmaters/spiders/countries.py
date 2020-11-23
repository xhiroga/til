import scrapy


class CountriesSpider(scrapy.Spider):
    name = 'countries'
    allowed_domains = ['www.worldmeters.info/']     # only domain. no protocol, no path
    start_urls = ['https://www.worldometers.info/world-population/poplulation-by-country/']  # fix to https

    def parse(self, response):
        pass
