from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!!'


@app.route('/data', methods=['GET'])
def get_data():
    return 'Some data'

