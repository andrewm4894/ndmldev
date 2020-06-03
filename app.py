import logging
import time
import asyncio
import async_timeout

import aiohttp
from flask import Flask, request, render_template, jsonify
from utils import get_chart_df, get_chart_list, parse_params
from ks import do_ks, rank_results


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=logging.INFO)


async def fetch(url):
    async with aiohttp.ClientSession() as session, async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()


def fight(responses):
    print(responses)
    return 'hello'


@app.route("/tmp")
def tmp():
    # perform multiple async requests concurrently
    responses = loop.run_until_complete(asyncio.gather(
        fetch("https://google.com/"),
        fetch("https://bing.com/"),
        fetch("https://duckduckgo.com"),
        fetch("http://www.dogpile.com"),
    ))

    # do something with the results
    return fight(responses)


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
