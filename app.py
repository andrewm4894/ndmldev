import logging
import time

from flask import Flask, request, render_template, jsonify
from pyod.models.knn import KNN

from get_data import get_data
from ks import do_ks
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

    chart_cols = {}
    for chart in charts:
        chart_cols[chart] = [colnames.index(col) for col in colnames if col.startswith(chart)]

    for chart in chart_cols:
        print('------------')
        print(chart)
        print(arr_baseline[:, chart_cols[chart]])
        print(arr_baseline[:, chart_cols[chart]])
        model = KNN(contamination=0.1, n_neighbors=5)
        model.fit(arr_baseline[:, chart_cols[chart]])
        anomaly_preds = model.predict(arr_highlight[:, chart_cols[chart]])
        anomaly_probs = model.predict_proba(arr_highlight[:, chart_cols[chart]])[:, 1]
        print('############')
        print(anomaly_preds)
        print(anomaly_probs)
        print('------------')

    XXX

    # do ks
    results = do_ks(colnames, arr_baseline, arr_highlight)
    time_got_ks = time.time()
    app.logger.info(f'... time data to ks = {round(time_got_ks - time_got_data, 2)}')

    # df_results
    df_results, df_results_chart = results_to_df(results, rank_by, rank_asc)
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
                    "data_host": "http://" + f"{remote_host.replace('127.0.0.1', local_host)}/".replace('//', '/')
                }
            )
        return render_template('results.html', charts=charts)
    elif response_format == 'json':
        return jsonify(df_results_chart.to_dict(orient='records'))
    else:
        return None
