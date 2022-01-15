from flask import Flask, request
app = Flask(__name__)
app.debug=True

@app.route('/')
def show_request_headers():
    return str(request.headers)

if __name__ == "__main__":
    app.run()
