import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule, CrawlSpider

title_top_250 = 'https://www.imdb.com/search/title/?groups=top_250&sort=user_rating/'


class BestMoviesSpider(CrawlSpider):  # CrawlSpider is extend class from Spider and has rule
    name = 'best_movies'
    allowed_domains = ['imdb.com']
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36'

    # instead of self.start_urls
    def start_requests(self):
        yield scrapy.Request(url=title_top_250, headers={
            'User-Agent': self.user_agent
        })

    rules = {
        Rule(LinkExtractor(restrict_xpaths="//h3[@class='lister-item-header']/a"), callback='parse_item', follow=True,
             process_request='set_user_agent'),
        Rule(LinkExtractor(restrict_xpaths="//a[@class='lister-page-next next-page']"))  # 自動的に2回目を呼ぶっぽい
    }

    def set_user_agent(self, request, spider):
        request.headers['User-Agent'] = self.user_agent
        return request

    # In this example, callback is overridden by parse_item
    def parse(self, response, **kwargs):
        pass

    def parse_item(self, response):
        yield {
            'title': response.xpath("//div[@class='title_wrapper']/h1/text()").get(),
            'year': response.xpath("//span[@id='titleYear']/a[1]/text()").get(),
            'duration': response.xpath("normalize-space(//div[@class='subtext']/time[1]/text())").get(),
            'genre': response.xpath("//div[@class='subtext']/a[2]/text()").get(),
            'rating': response.xpath("normalize-space(//div[@class='subtext']/text())").get(),
            'movie_url': response.url,
            'user-agent': response.request.headers['User-Agent']
        }
