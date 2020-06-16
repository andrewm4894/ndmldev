import logging
import time
from collections import OrderedDict, Counter

from flask import Flask, request, render_template, jsonify
import pandas as pd

from netdata_pandas.data import get_data
from model import run_model
from utils import get_chart_list, parse_params, results_to_df


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=logging.INFO)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results')
def results():

    time_start = time.time()

    # get params
    params = parse_params(request)
    app.logger.info(f'... params = {params}')
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    return_type = params['return_type']
    remote_host = params['remote_host']
    local_host = params['local_host']
    model = params['model']
    score_thold = params['score_thold']
    model_level = model.get('model_level', 'dimension')

    # get charts to pull data for
    charts = get_chart_list(host=remote_host)

    # get data
    df = get_data(remote_host, charts, after=baseline_after, before=highlight_before,
                  diff=True, ffill=True, numeric_only=True, nunique_thold=0.05)
    colnames = list(df.columns)
    arr_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').values
    arr_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').values
    charts = list(set([col.split('|')[0] for col in colnames]))
    app.logger.info(f'... len(charts) = {len(charts)}')
    app.logger.info(f'... len(colnames) = {len(colnames)}')
    app.logger.info(f'... arr_baseline.shape = {arr_baseline.shape}')
    app.logger.info(f'... arr_highlight.shape = {arr_highlight.shape}')
    time_got_data = time.time()
    app.logger.info(f'... time start to data = {time_got_data - time_start}')

    # get scores
    results_dict = run_model(model, colnames, arr_baseline, arr_highlight)

    time_got_scores = time.time()
    app.logger.info(f'... time data to scores = {round(time_got_scores - time_got_data, 2)}')

    # get max and min scores
    scores = []
    for chart in results_dict:
        for dimension in results_dict[chart]:
            scores.append([k['score'] for k in dimension.values()])
    score_max = max(scores)[0]
    score_min = min(scores)[0]

    # normalize scores
    results_list = []
    for chart in results_dict:
        for dimension in results_dict[chart]:
            for k in dimension:
                score = dimension[k]['score']
                score_norm = (score - score_min)/(score_max - score_min)
                results_list.append([chart, k, score, score_norm])

    df_results = pd.DataFrame(results_list, columns=['chart', 'dimension', 'score', 'score_norm'])
    if score_thold > 0:
        df_results = df_results[df_results['score_norm'] >= score_thold]
    df_results['rank'] = df_results['score'].rank(method='first', ascending=False)

    time_done = time.time()
    app.logger.info(f'... time total = {round(time_done - time_start, 2)}')

    #print(df_results)
    for chart in df_results['chart'].unique():
        print(chart)
        print(','.join(df_results[df_results['chart'] == chart]['dimension'].values.tolist()))
    #XXX

    # df_results_chart
    #df_results_chart = results_to_df(results, model)
    #if score_thold > 0:
    #    df_results_chart = df_results_chart[df_results_chart['score'] >= score_thold]
    #time_got_results = time.time()
    #app.logger.info(f'... time scores to results = {round(time_got_results - time_got_scores, 2)}')

    #print(df_results_chart)
    #xxx



    # build response
    if return_type == 'html':
        charts = df_results['chart'].values.tolist()
        counts = OrderedDict(Counter([c.split('.')[0] for c in charts]).most_common())
        counts = ' | '.join([f"{c}:{counts[c]}" for c in counts])
        summary_text = f"number of charts = {df_results['chart'].nunique()}, number of dimensions = {len(df_results)}, {counts}"
        charts_to_render = []
        for chart in df_results['chart'].unique():
            dimensions = ','.join(df_results[df_results['chart'] == chart]['dimension'].values.tolist())
            charts_to_render.append(
                {
                    "id": chart,
                    "title": chart,
                    "after": baseline_after,
                    "before": highlight_before,
                    "data_host": "http://" + f"{remote_host.replace('127.0.0.1', local_host)}/".replace('//', '/'),
                    "dimensions": dimensions
                }
            )
        return render_template(
            'results.html', charts=charts_to_render, highlight_after=highlight_after*1000,
            highlight_before=highlight_before*1000, summary_text=summary_text
        )
    elif return_type == 'json':
        return jsonify(df_results.to_dict(orient='records'))
    else:
        return None
