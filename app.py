from datetime import datetime, timedelta

from flask import Flask, request
import pandas as pd
from utils import get_chart_data_urls, get_chart_df, do_ks, get_chart_list

app = Flask(__name__)


@app.route('/ks')
def ks():
    before = int(datetime.now().timestamp())
    after = before - 500
    window_end = before - 10
    window_start = window_end - 50
    baseline_end = window_start - 1
    baseline_start = baseline_end - 100
    results = {}
    for chart in get_chart_list():
        print(chart)
        df = get_chart_df(chart, after, before)
        print(df.shape)
        print(df.head())
        if len(df) > 0:
            ks_results = do_ks(
                df,
                baseline_start,
                baseline_end,
                window_start,
                window_end
            )
            if ks_results:
                results[chart] = ks_results
    df_rank = pd.DataFrame(
        data=[[c, results[c]['summary']['ks_mean']] for c in results],
        columns=['chart', 'score']
    )
    df_rank['rank'] = df_rank['score'].rank()
    print(df_rank)
    for _, row in df_rank.iterrows():
        results[row['chart']]['rank'] = int(row['rank'])
    print(results)
    return results


@app.route('/')
def home():
    response = dict(
        ks='/ks'
    )
    return response


@app.route('/dash')
def dash():
    dash_template = open('templates/dash-example.html', 'r', encoding='utf-8')
    dash_template_html = dash_template.read()
    dash_file_out = open("/usr/share/netdata/web/dash2.html", "w+")
    dash_file_out.write(dash_template_html)
    dash_file_out.close()
    response = dict(
        message='done'
    )
    return response


@app.route('/tmp', methods=['GET'])
def xdo_ks():
    raw_query_string = request.query_string.decode()
    print(raw_query_string)
    now = datetime.now()
    before = request.args.get('before', int(now.timestamp()))
    after = request.args.get('after', int((now-timedelta(seconds=100)).timestamp()))
    highlight_before = request.args.get('highlight_before', int(now.timestamp()))
    highlight_after = request.args.get('highlight_after', int((now-timedelta(seconds=100)).timestamp()))
    response = {
        "info": dict(
            before=before,
            after=after,
            highlight_before=highlight_before,
            highlight_after=highlight_after
        ),
        "results": {
        }
    }
    return response

