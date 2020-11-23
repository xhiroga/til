import scrapy


class CountriesSpider(scrapy.Spider):
    name = 'countries'
    allowed_domains = ['www.worldometers.info']  # only domain. no protocol, no path, no end slash
    start_urls = ['https://www.worldometers.info/world-population/population-by-country/']  # fix to https

    def parse(self, response):
        countries = response.xpath("//td/a")
        for country in countries:
            name = country.xpath(".//text()").get()  # Selector オブジェクトに対して実行する場合は .// から始める。
            link = country.xpath(".//@href").get()

            # absolute_url = response.url_join(link)
            # OR
            # absolute_url = f"https://worldmeters.info{link}"
            # yield scrapy.Request(absolute_url)
            # is equivalent to
            yield response.follow(url=link, callback=self.parse_country, meta={'country_name': name})

            # yield scrapy.Request(link) raise ValueError(f'Missing scheme in request url: {self._url}')

    def parse_country(self, response):
        # inspect_response(self, response)  # REPLでレスポンスオブジェクトを参照
        # open_in_browser(response)         # レスポンスをローカルに保存して表示
        name = response.request.meta['country_name']
        rows = response.xpath(
            "//table[@class='table table-striped table-bordered table-hover table-condensed table-list'][1]/tbody/tr")
        for row in rows:
            year = row.xpath(".//td[1]/text()").get()
            population = row.xpath('.//td[2]/strong/text()').get()
            yield {
                'name': name,
                'year': year,
                'population': population
            }
