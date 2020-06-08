from datetime import datetime
from urllib.parse import parse_qs, urlparse

import requests
import pandas as pd


def get_chart_list(starts_with: str = None, host: str = '127.0.0.1:19999'):
    url = f"http://{host}/api/v1/charts"
    print(url)
    XXX
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_list = [chart for chart in charts]
    if starts_with:
        chart_list = [chart for chart in chart_list if chart.startswith(starts_with)]
    return chart_list


def filter_useless_cols(df):
    s = (df.min() == df.max())
    useless_cols = list(s.where(s == True).dropna().index)
    df = df.drop(useless_cols, axis=1)
    return df


def parse_params(request):

    now = int(datetime.now().timestamp())
    default_window_size = 60 * 2
    baseline_window_multiplier = 2

    url_parse = urlparse(request.args.get('url'))
    url_params = parse_qs(request.args.get('url'))
    remote_host = url_parse.netloc.split(':')[0]
    if remote_host == request.host.split(':')[0]:
        remote_host = '127.0.0.1:19999'
    local_host = request.host.split(':')[0]
    if 'after' in url_params:
        after = int(int(url_params.get('after')[0]) / 1000)
    else:
        after = int(request.args.get('after', now - default_window_size))
    if 'before' in url_params:
        before = int(int(url_params.get('before')[0]) / 1000)
    else:
        before = int(request.args.get('before', now))
    if 'highlight_after' in url_params:
        highlight_after = int(int(url_params.get('highlight_after')[0]) / 1000)
    else:
        highlight_after = request.args.get('highlight_after', after)
    if 'highlight_before' in url_params:
        highlight_before = int(int(url_params.get('highlight_before')[0]) / 1000)
    else:
        highlight_before = request.args.get('highlight_before', before)

    if url_parse.path.startswith('/host/'):
        remote_host = f'{remote_host}{url_parse.path[:-1]}'
        local_host = f'{local_host}{url_parse.path[:-1]}'

    print(remote_host)
    print(local_host)

    window_size = highlight_before - highlight_after
    baseline_before = highlight_after - 1
    baseline_after = baseline_before - (window_size * baseline_window_multiplier)
    params = {
        "before": before,
        "after": after,
        "highlight_before": highlight_before,
        "highlight_after": highlight_after,
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "rank_by": request.args.get('rank_by', 'ks_max'),
        "rank_asc": request.args.get('rank_asc', False),
        "starts_with": request.args.get('starts_with', None),
        "format": request.args.get('format', 'html'),
        "remote_host": remote_host,
        "local_host": local_host,
        "method": request.args.get('method', 'ks'),
        "run_mode": request.args.get('run_mode', 'default')
    }
    return params


def results_to_df(results, rank_by, rank_asc):

    rank_by_var = rank_by.split('_')[0]

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

    return df_results, df_results_chart

