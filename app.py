from flask import Flask, request, render_template, jsonify
from utils import get_chart_df, get_chart_list, parse_params
from ks import do_ks, rank_results

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results')
def results():

    # get params
    params = parse_params(request)
    highlight_before = params['highlight_before']
    highlight_after = params['highlight_after']
    baseline_before = params['baseline_before']
    baseline_after = params['baseline_after']
    rank_by = params['rank_by']
    starts_with = params['starts_with']
    response_format = params['format']
    netdata_host = params['netdata_host']

    # get results
    results = {}
    for chart in get_chart_list(starts_with=starts_with):
        df = get_chart_df(chart, after=baseline_after, before=highlight_before, host=netdata_host)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results
    results = rank_results(results, rank_by, ascending=False)

    if response_format == 'html':
        charts = [
            {
                "id": result,
                "title": f"{results[result]['rank']} - {result} (ks={results[result]['summary']['ks_mean']}, p={results[result]['summary']['p_mean']})",
                "after": baseline_after,
                "before": highlight_before
            } for result in results
        ]
        return render_template('results.html', charts=charts)
    else:
        return jsonify(results)
