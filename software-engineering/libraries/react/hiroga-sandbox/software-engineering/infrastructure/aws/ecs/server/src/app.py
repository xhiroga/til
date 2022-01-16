from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Flask server on Docker works!'

if __name__ == '__main__':
    print('app run!')
    app.run(host='0.0.0.0', port=5000, debug=True)