from datetime import datetime, timedelta

from flask import Flask, request, render_template
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
    dash_template = open('templates/results_dashboard.html', 'r', encoding='utf-8')
    dash_template_html = dash_template.read()
    dash_file_out = open("/usr/share/netdata/web/results_dashboard.html", "w+")
    dash_file_out.write(dash_template_html)
    dash_file_out.close()
    response = dict(
        message='done'
    )
    charts = [
        {"id": "system.cpu", "title": "cpu"},
        {"id": "system.load", "title": "load"},
        {"id": "system.io", "title": "io"},
    ]
    return render_template('results_dashboard.html', charts=charts)


@app.route('/tmp', methods=['GET'])
def tmp():
    default_window_size = 60
    default_baseline_window_multiplier = 1
    now = int(datetime.now().timestamp())
    url = request.args.get('url', None)
    if url:
        url_parts = url.split(';')
        after = [x.split('=')[1] for x in url_parts if x.startswith('after=')][0]
        print(after)
    before = request.args.get('before', now)
    after = request.args.get('after', now - default_window_size)
    highlight_before = request.args.get('highlight_before', now)
    highlight_after = request.args.get('highlight_after', now - default_window_size)
    window_multiplier = request.args.get('window_multiplier', default_baseline_window_multiplier)
    window_size = highlight_before - highlight_after
    baseline_before = highlight_after - 1
    baseline_after = baseline_before - (window_size * window_multiplier)
    response = {
        "info": dict(
            url=url,
            before=before,
            after=after,
            highlight_before=highlight_before,
            highlight_after=highlight_after,
            baseline_before=baseline_before,
            baseline_after=baseline_after
        ),
        "results": {
        }
    }
    return response

