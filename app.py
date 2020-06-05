import logging
import time

import trio
from flask import Flask, request, render_template, jsonify
from scipy.stats import ks_2samp

from get_data import get_charts_df_async, get_data
from utils import get_chart_list, parse_params
import pandas as pd


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
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    rank_by = params['rank_by']
    rank_asc = params['rank_asc']
    starts_with = params['starts_with']
    response_format = params['format']
    remote_host = params['remote_host']
    local_host = params['local_host']
    rank_by_var = rank_by.split('_')[0]

    # get charts to pull data for
    charts = get_chart_list(starts_with=starts_with, host=remote_host)
    arr_baseline, arr_highlight = get_data(remote_host, charts, baseline_after, baseline_before, highlight_after,
                                           highlight_before)
    time_got_data = time.time()
    app.logger.info(f'... time start to data = {time_got_data - time_start}')

    # get ks
    results = []
    for n in range(arr_baseline.shape[1]):
        ks_stat, p_value = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        results.append([ks_stat, p_value])
    time_got_ks = time.time()
    app.logger.info(f'... time data to ks = {round(time_got_ks - time_got_data, 2)}')

    # wrangle results
    results = zip([[col.split('__')[0], col.split('__')[1]] for col in list(df.columns)], results)
    results = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in results]

    # df_results
    df_results = pd.DataFrame(results, columns=['chart', 'dimension', 'ks', 'p'])
    df_results['rank'] = df_results[rank_by_var].rank(method='first', ascending=rank_asc)
    df_results = df_results.sort_values('rank')

    # df_results_chart
    df_results_chart = df_results.groupby(['chart'])[['ks', 'p']].agg(['mean', 'min', 'max'])
    df_results_chart.columns = ['_'.join(col) for col in df_results_chart.columns]
    df_results_chart = df_results_chart.reset_index()
    df_results_chart['rank'] = df_results_chart[rank_by].rank(method='first', ascending=rank_asc)
    df_results_chart = df_results_chart.sort_values('rank')

    time_got_results = time.time()
    app.logger.info(f'... time ks to results = {round(time_got_results - time_got_ks, 2)}')

    time_done = time.time()
    app.logger.info(f'... time total = {round(time_done - time_start, 2)}')

    # build response
    if response_format == 'html':
        charts = []
        for i, row in df_results_chart.iterrows():
            charts.append(
                {
                    "id": row['chart'],
                    "title": f"{row['rank']} - {row['chart']} (ks={round(row[rank_by],2)}, p={round(row['p_min'],2)})",
                    "after": baseline_after,
                    "before": highlight_before,
                    "data_host": f"http://{remote_host.replace('127.0.0.1', local_host)}/"
                }
            )
        return render_template('results.html', charts=charts)
    elif response_format == 'json':
        return jsonify(df_results_chart.to_dict(orient='records'))
    else:
        return None
