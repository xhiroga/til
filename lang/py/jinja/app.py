from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html', name="Hiroaki", user={"mail": "example.com", "tell": "00-0000-0000"})


if __name__ == "__main__":
    app.debug = True
    app.run(port=3000)
