import tornado.httpserver
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


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/register", RegisterHandler),
        (r"/thanks", ThanksHandler),
    ])


if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(make_app())
    http_server.listen(tornado.options.options.port)

    tornado.ioloop.IOLoop.current().start()
