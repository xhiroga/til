import os
from jinja2 import Environment, FileSystemLoader
import tornado.ioloop
import tornado.web

from tornado.options import define
define("port", default=5000, help="run on the given port", type=int)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        template = env.get_template('./login.html')
        self.write(template.render())


class RegisterHandler(tornado.web.RequestHandler):
    def post(self):
        # ここで登録処理をする。
        self.redirect(r"/thanks")


class ThanksHandler(tornado.web.RequestHandler):
    def get(self):
        # 再読み込みされるのはこのページのため、登録処理が二重に行われることがない。
        self.write("Thanks! You are redirected!!")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/register", RegisterHandler),
        (r"/thanks", ThanksHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(int(os.environ.get("PORT", 5000)))
    print("server is listening in port 5000!")

    tornado.ioloop.IOLoop.current().start()
