from datetime import datetime, timedelta

from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!!'


@app.route('/ks', methods=['GET'])
def get_data():
    now = datetime.now()
    context_chart = request.args.get('context_chart', 'system.cpu')
    window_start = request.args.get('context_chart', now)
    window_end = request.args.get('context_chart', now - timedelta(minutes=1))
    response = {
        "info": dict(
            context_chart=context_chart,
            window_start=window_start,
            window_end=window_end
        ),
        "results": {
            1: {"chart_name": "blah", "score": 0.33, "p": 0.01, "rank": 1},
            2: {"chart_name": "foo", "score": 0.2, "p": 0.05, "rank": 2}
        }
    }
    return response

