# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__) #ローカルポート5000番
CORS(app)

@app.route("/", methods=['GET'])
def callback():
    return render_template('index.html')


if __name__ == "__main__":
    app.debug = True;
    print("server run!")
    app.run()
