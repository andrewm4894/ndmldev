from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!!'


@app.route('/ks', methods=['GET'])
def get_data():
    response = {
        "results": {
            1: {"chart_name": "blah", "score": 0.33, "p": 0.01, "rank": 1},
            2: {"chart_name": "foo", "score": 0.2, "p": 0.05, "rank": 2}
        }
    }
    return response

