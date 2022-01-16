import os
from jinja2 import Environment, FileSystemLoader
import tornado.ioloop
import tornado.web


from tornado.options import define
define("port", default=5000, help="run on the given port", type=int)


class MagicHandler(tornado.web.RequestHandler):
    def get(self):
        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        template = env.get_template('./magic.html')
        self.write(template.render())


class AjaxHandler(tornado.web.RequestHandler):
    def get(self):
        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        template = env.get_template('./ajax.html')
        self.write(template.render())


class MultipleHandler(tornado.web.RequestHandler):
    def get(self):
        print("/multiple ACCESS!!")
        self.write(str(int(self.get_argument("input_number"))*5))


def make_app():
    return tornado.web.Application([
        (r"/", MagicHandler),
        (r"/ajax", AjaxHandler),
        (r"/multiple", MultipleHandler),
        (r"/magic", MagicHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "./static/"})
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(int(os.environ.get("PORT", 5000)))
    print("server is listening in port 5000!")

    tornado.ioloop.IOLoop.current().start()
