import logging
import time

from flask import Flask, request, render_template, jsonify

from data import get_data
from ml import do_ks, do_pyod
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
    method = params['method']

    # get charts to pull data for
    charts = get_chart_list(starts_with=starts_with, host=remote_host)

    # get data
    colnames, arr_baseline, arr_highlight = get_data(
        remote_host, charts, baseline_after, baseline_before, highlight_after, highlight_before
    )
    charts = list(set([col.split('__')[0] for col in colnames]))
    app.logger.info(f'... len(charts) = {len(charts)}')
    app.logger.info(f'... len(colnames) = {len(colnames)}')
    app.logger.info(f'... arr_baseline.shape = {arr_baseline.shape}')
    app.logger.info(f'... arr_highlight.shape = {arr_highlight.shape}')
    time_got_data = time.time()
    app.logger.info(f'... time start to data = {time_got_data - time_start}')

    # get scores
    if method == 'pyod':
        chart_cols = {}
        for chart in charts:
            chart_cols[chart] = [colnames.index(col) for col in colnames if col.startswith(chart)]
        results = do_pyod(chart_cols, arr_baseline, arr_highlight)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)

    time_got_scores = time.time()
    app.logger.info(f'... time data to scores = {round(time_got_scores - time_got_data, 2)}')

    # df_results_chart
    df_results_chart = results_to_df(results, rank_by, rank_asc, method=method)
    time_got_results = time.time()
    app.logger.info(f'... time scores to results = {round(time_got_results - time_got_scores, 2)}')

    time_done = time.time()
    app.logger.info(f'... time total = {round(time_done - time_start, 2)}')

    # build response
    if response_format == 'html':
        charts = []
        for i, row in df_results_chart.iterrows():
            if 1 == 1:
                title = f"{row.to_json()}"
            elif method == 'pyod':
                title = f"{row['rank']} - {row['chart']} (pred={round(row['pred'],2)}, prob={round(row['prob'],2)})"
            else:
                title = f"{row['rank']} - {row['chart']} (ks={round(row[rank_by],2)}, p={round(row['p_min'],2)})"
            charts.append(
                {
                    "id": row['chart'],
                    "title": title,
                    "after": baseline_after,
                    "before": highlight_before,
                    "data_host": "http://" + f"{remote_host.replace('127.0.0.1', local_host)}/".replace('//', '/')
                }
            )
        return render_template('results.html', charts=charts)
    elif response_format == 'json':
        return jsonify(df_results_chart.to_dict(orient='records'))
    else:
        return None
