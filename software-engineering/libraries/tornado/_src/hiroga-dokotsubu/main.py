import tornado.ioloop
import tornado.web

from jinja2 import Environment, FileSystemLoader


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        # self.write("Hello, world")

        book = {
            'title': '陽だまりの彼女',
            'author': '越谷オサム',
            'description': '二人の生活感ある日常が魅力'
        }

        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        template = env.get_template('./index.html')
        html = template.render({'book': book})
        self.write(html.encode('utf-8'))

    def post(self):
        print(self.get_body_argument('tsubuyaki'))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
    print("main start!")
