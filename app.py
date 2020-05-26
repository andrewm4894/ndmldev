from datetime import datetime, timedelta

from flask import Flask, request, render_template
import pandas as pd
from utils import get_chart_data_urls, get_chart_df, do_ks, get_chart_list, parse_params

app = Flask(__name__)


@app.route('/ks')
def ks():
    params = parse_params(request)
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    results = {}
    for chart in get_chart_list(starts_with='system.'):
        df = get_chart_df(chart, baseline_after, highlight_before)
        if len(df) > 0:
            ks_results = do_ks(
                df,
                baseline_after,
                baseline_before,
                highlight_after,
                highlight_before
            )
            if ks_results:
                results[chart] = ks_results
    df_rank = pd.DataFrame(
        data=[[c, results[c]['summary']['ks_mean']] for c in results],
        columns=['chart', 'score']
    )
    df_rank['rank'] = df_rank['score'].rank()
    for _, row in df_rank.iterrows():
        results[row['chart']]['rank'] = int(row['rank'])
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
    params = parse_params(request)
    response = {
        "params": params,
        "results": {
        }
    }
    return response

