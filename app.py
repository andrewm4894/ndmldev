import json
from collections import OrderedDict
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

from flask import Flask, request, render_template, session, jsonify
import pandas as pd
from utils import get_chart_df, get_chart_list, parse_params
from ks import do_ks


app = Flask(__name__)


@app.route('/ks')
def ks():
    params = parse_params(request)
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    rank_by = params['rank_by']
    starts_with = params['starts_with']
    response_format = params['format']
    results = {}
    for chart in get_chart_list(starts_with=starts_with):
        df = get_chart_df(chart, baseline_after, highlight_before)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results
    df_rank = pd.DataFrame(data=[[c, results[c]['summary'][rank_by]] for c in results], columns=['chart', 'score'])
    df_rank['rank'] = df_rank['score'].rank(method='first')
    for _, row in df_rank.iterrows():
        results[row['chart']]['rank'] = int(row['rank'])
        results[row['chart']]['score'] = float(row['score'])
    results = OrderedDict(sorted(results.items(), key=lambda t: t[1]["rank"]))
    print(results)
    if response_format == 'json':
        return json.dumps([results])
    elif response_format == 'html':
        max_rank = 50
        render_info = []
        for chart in results:
            rank = results[chart]['rank']
            if rank >= max_rank:
                break
            else:
                score = results[chart]['score']
                render_info.append({
                    "chart_id": chart,
                    "title": f"{rank} - {chart} - {score}",
                    "before": baseline_after,
                    "after": highlight_before,
                })
        return render_template('results.html', render_info=render_info)
    else:
        return None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results')
def results():

    # params
    #now_ts = int(datetime.now().timestamp())
    #netdata_url = request.args.get('url')
    #netdata_host = urlparse(netdata_url).netloc
    #netdata_params = parse_qs(netdata_url)
    #after = int(netdata_params.get('after')[0])
    #before = int(netdata_params.get('before')[0])
    #highlight_after = int(netdata_params.get('highlight_after')[0])
    #highlight_before = int(netdata_params.get('highlight_before')[0])
    #after_secs = str(after / 1000)
    #before_secs = str(before / 1000)

    # get params
    params = parse_params(request)
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    rank_by = params['rank_by']
    starts_with = params['starts_with']
    response_format = params['format']

    # get results
    results = {}
    for chart in get_chart_list(starts_with=starts_with):
        df = get_chart_df(chart, baseline_after, highlight_before)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results
    df_rank = pd.DataFrame(data=[[c, results[c]['summary'][rank_by]] for c in results], columns=['chart', 'score'])
    df_rank['rank'] = df_rank['score'].rank(method='first')
    for _, row in df_rank.iterrows():
        results[row['chart']]['rank'] = int(row['rank'])
        results[row['chart']]['score'] = float(row['score'])
    results = OrderedDict(sorted(results.items(), key=lambda t: t[1]["rank"]))
    print(results)
    XXX

    charts = [
        {"id": "system.cpu", "title": "cpu", "after": after_secs, "before": before_secs},
        {"id": "system.load", "title": "load", "after": after_secs, "before": before_secs},
        {"id": "system.io", "title": "io", "after": after_secs, "before": before_secs},
    ]
    return render_template('results.html', charts=charts)


@app.route('/dash')
def dash():
    results = session.get('results', None)
    print(results)
    dash_template = open('templates/results.html', 'r', encoding='utf-8')
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
    return render_template('results.html', charts=charts)


@app.route('/tmp', methods=['GET'])
def tmp():
    params = parse_params(request)
    response = {
        "params": params,
        "results": {
        }
    }
    return response

