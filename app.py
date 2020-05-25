from datetime import datetime, timedelta

from flask import Flask, request
from utils import get_chart_urls

app = Flask(__name__)


@app.route('/')
def home_info():
    response = dict(
        ks='/ks'
    )
    return response


@app.route('/tmp')
def tmp():

    response = dict(
        ks='/ks'
    )
    return response


@app.route('/ks', methods=['GET'])
def do_ks():
    now = datetime.now()
    context_chart = request.args.get('context_chart', 'system.cpu')
    window_start = request.args.get('window_start', now)
    window_end = request.args.get('window_end', now - timedelta(minutes=1))
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

