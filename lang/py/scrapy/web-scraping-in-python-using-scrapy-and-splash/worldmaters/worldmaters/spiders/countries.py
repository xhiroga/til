import scrapy


class CountriesSpider(scrapy.Spider):
    name = 'countries'
    allowed_domains = ['www.worldmeters.info/']     # only domain. no protocol, no path
    start_urls = ['https://www.worldometers.info/world-population/population-by-country/']  # fix to https

    def parse(self, response):
        countries = response.xpath("//td/a")
        for country in countries:
            name = country.xpath(".//text()").get()     # Selecter オブジェクトに対して実行する場合は .// から始める。
            link = country.xpath(".//@href").get()

            absolute_url = f"https://worldmeters.info{link}"
            yield scrapy.Request(absolute_url)
            # yield scrapy.Request(link) raise ValueError(f'Missing scheme in request url: {self._url}')
