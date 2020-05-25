from datetime import datetime, timedelta

from flask import Flask, request
import pandas as pd
from utils import get_chart_data_urls, get_chart_df, do_ks

app = Flask(__name__)


@app.route('/tmp')
def tmp():
    before = int(datetime.now().timestamp())
    after = before - 500
    window_end = before - 10
    window_start = window_end - 50
    baseline_end = window_start - 1
    baseline_start = baseline_end - 100
    results = {}
    for chart in ['system.cpu', 'system.load']:
        results[chart] = do_ks(
            get_chart_df(chart, after, before),
            baseline_start,
            baseline_end,
            window_start,
            window_end
        )
    df_rank = pd.DataFrame(
        data=[[c, results[c]['summary']['ks_mean']] for c in results],
        columns=['chart', 'score']
    )
    df_rank['rank'] = df_rank['score'].rank()
    for _, row in df_rank.iterrows():
        print(row['chart'], row['rank'])
    return results


@app.route('/')
def home_info():
    response = dict(
        ks='/ks'
    )
    return response


@app.route('/ks', methods=['GET'])
def xdo_ks():
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

