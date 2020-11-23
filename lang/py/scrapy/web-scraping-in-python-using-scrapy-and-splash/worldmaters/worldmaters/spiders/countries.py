import scrapy


class CountriesSpider(scrapy.Spider):
    name = 'countries'
    allowed_domains = ['www.worldmeters.info/']     # only domain. no protocol, no path
    start_urls = ['https://www.worldometers.info/world-population/population-by-country/']  # fix to https

    def parse(self, response):
        title = response.xpath("//h1/text()").get()
        countries = response.xpath("//td/a/text()").getall()

        yield{
            'title': title,
            'countries': countries
        }
