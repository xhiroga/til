import scrapy
from scrapy_splash import SplashRequest, SplashFormRequest


class QuotesLoginSplashSpider(scrapy.Spider):
    name = 'quotes_login_splash'
    allowed_domains = ['quotes.toscrape.com']

    script = '''
        function main(splash, args)
          assert(splash:go(args.url))
          assert(splash:wait(0.5))
          return splash:html()
        end
    '''

    def start_requests(self):
        yield SplashRequest(
            url='https://quotes.toscrape.com/login',
            endpoint='execute',
            args={
                'lua_source': self.script
            },
            callback=self.parse
        )

    def parse(self, response):
        csrf_token = response.xpath('//input[@name="csrf_token"]/@value').get()
        yield SplashFormRequest.from_response(
            response,
            formxpath='//form',
            formdata={
                'csrf_token': csrf_token,
                'username': 'admin',
                'password': 'admin'
            },
            callback=self.after_login
        )

    def after_login(self, response):
        if response.xpath("//a[@href='/logout']/text()").get():
            print('logged in')
