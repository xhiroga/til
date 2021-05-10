import tornado.ioloop
import tornado.web

# ログイン済なら"Hello, {名前}"と表示し、そうでなければログインページに遷移させる。
# 普通ユーザーの入力値をCookieに保存しないと思う。あくまで例。


class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        return self.get_cookie("user")


class MainHandler(BaseHandler):
    def get(self):
        if not self.current_user:
            self.redirect("/login")
            return
        name = tornado.escape.xhtml_escape(self.current_user)
        self.write("Hello, " + name)


class LoginHandler(BaseHandler):
    def get(self):
        self.write('<html><body><form action="/login" method="post">'
            'Name: <input type="text" name="name">'
            '<input type="submit" value="Sign in">'
            '</form></body></html>')
    def post(self):
        self.set_cookie("user", self.get_argument("name"))
        self.redirect("/")


if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/login", LoginHandler),
    ])
    app.listen(8888)
    print("Start listen...")
    tornado.ioloop.IOLoop.current().start()

# 参考
# https://sites.google.com/site/tornadowebja/documentation/overview
