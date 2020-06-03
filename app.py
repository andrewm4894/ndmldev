import logging
import time
from io import BytesIO

import trio
import asks
from flask import Flask, request, render_template, jsonify
from utils import get_chart_df, get_chart_list, parse_params
from ks import do_ks, rank_results
import pandas as pd



app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=logging.INFO)


async def fetch(api_call, results):
    url, chart = api_call
    response = await asks.get(url)
    df = pd.read_csv(BytesIO(response.content)).set_index('time').add_prefix('{}.'.format(chart))
    results[chart] = df


async def fetch_all(api_calls):
    results = {}
    with trio.move_on_after(5):
        async with trio.open_nursery() as nursery:
            for api_call in api_calls:
                nursery.start_soon(fetch, api_call, results)
    df = pd.concat(results, join='outer', axis=1, sort=True)
    print(df.shape)
    print(df.head())


@app.route("/tmp")
def tmp():
    api_calls = [
        ("http://london.my-netdata.io/api/v1/data?chart=system.cpu&format=csv", "system.cpu"),
        ("http://london.my-netdata.io/api/v1/data?chart=system.load&format=csv", "system.load")
    ]
    trio.run(fetch_all(), api_calls)
    return 'hello'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results')
def results():

    start_time = time.time()

    # get params
    params = parse_params(request)
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    rank_by = params['rank_by']
    starts_with = params['starts_with']
    response_format = params['format']
    remote_host = params['remote_host']
    local_host = params['local_host']

    # get results
    results = {}
    for chart in get_chart_list(starts_with=starts_with, host=remote_host):
        df = get_chart_df(chart, after=baseline_after, before=highlight_before, host=remote_host)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results
    results = rank_results(results, rank_by, ascending=False)

    app.logger.info(f"time taken = {time.time()-start_time}")

    # build response
    if response_format == 'html':
        charts = [
            {
                "id": result,
                "title": f"{results[result]['rank']} - {result} (ks={results[result]['summary']['ks_max']}, p={results[result]['summary']['p_min']})",
                "after": baseline_after,
                "before": highlight_before,
                "data_host": f"http://{remote_host.replace('127.0.0.1', local_host)}/"

            } for result in results
        ]
        return render_template('results.html', charts=charts)
    else:
        return jsonify(results)
