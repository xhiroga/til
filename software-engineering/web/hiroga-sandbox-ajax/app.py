import os

import tornado.ioloop
import tornado.web
from jinja2 import Environment, FileSystemLoader

from tornado.options import define
define("port", default=5000, help="run on the given port", type=int)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        template = env.get_template('./login.html')
        self.write(template.render())


class RegisterHandler(tornado.web.RequestHandler):
    def post(self):
        # 登録処理
        self.redirect(r"/thanks")


class ThanksHandler(tornado.web.RequestHandler):
    def get(self):
        # 登録処理
        self.write("Thanks!!! You are redirected!!")


class AjaxHandler(tornado.web.RequestHandler):
    # 参考
    # https://www.ibm.com/developerworks/jp/web/library/wa-ajaxintro1.html
    def get(self):
        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        template = env.get_template('./ajax.html')
        self.write(template.render())


class MultipleHandler(tornado.web.RequestHandler):
    def get(self):
        print("/multiple ACCESS!!")
        self.write("*5")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/register", RegisterHandler),
        (r"/thanks", ThanksHandler),
        (r"/ajax", AjaxHandler),
        (r"/multiple", MultipleHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(int(os.environ.get("PORT", 5000)))

    tornado.ioloop.IOLoop.current().start()
