import json
import logging
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import requests
import pandas as pd

from model import supported_pyod_models

log = logging.getLogger(__name__)


def get_chart_list(starts_with: str = None, host: str = '127.0.0.1:19999'):
    url = f"http://{host}/api/v1/charts"
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_list = [chart for chart in charts]
    if starts_with:
        chart_list = [chart for chart in chart_list if chart.startswith(starts_with)]
    return chart_list


def filter_useless_cols(df, nunique_thold=0.05):
    s = (df.min() == df.max())
    useless_cols = list(s.where(s == True).dropna().index)
    df = df.drop(useless_cols, axis=1)
    df = df.dropna(axis=1)
    df = df.loc[:, df.nunique() / len(df) > nunique_thold]
    return df


def filter_lowstd_cols(df, std_thold=0.05):
    df = df.loc[:, df.std() > std_thold]
    return df


def parse_params(request):

    now = int(datetime.now().timestamp())
    default_window_size = 60 * 2
    default_baseline_window_multiplier = 2

    url_parse = urlparse(request.args.get('url'))
    url_params = parse_qs(request.args.get('url'))

    ks_config_default = {
        "model": {
            "type": "ks",
            "params": {},
            "n_lags": 0
        },
        "return_type": "html",
        "baseline_window_multiplier": 2,
        "score_thold": 0.2
    }

    pyod_config_default = {
        "model": {
            "type": "hbos",
            "params": {"contamination": 0.1},
            "n_lags": 2
        },
        "return_type": "html",
        "baseline_window_multiplier": 2,
        "score_thold": 0.2
    }

    config = json.loads(request.args.get('config', json.dumps(ks_config_default)))

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
        remote_host = f'{remote_host}:19999{url_parse.path[:-1]}'
        local_host = f'{local_host}:19999{url_parse.path[:-1]}'

    baseline_window_multiplier = config.get('baseline_window_multiplier', default_baseline_window_multiplier)
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
        "remote_host": remote_host,
        "local_host": local_host,
        "model": config.get('model'),
        "return_type": config.get('return_type', 'html'),
        "score_thold": config.get('score_thold', 0),
    }
    return params


def results_to_df(results, model):

    if model['type'] in supported_pyod_models:

        rank_asc = False

        # df_results_chart
        df_results_chart = pd.DataFrame(results, columns=['chart', 'prob', 'pred'])
        df_results_chart['score'] = (df_results_chart['prob'] + df_results_chart['pred']) / 2
        df_results_chart['rank'] = df_results_chart['score'].rank(method='first', ascending=rank_asc)
        df_results_chart = df_results_chart.sort_values('rank')

    elif model['type'] == 'ks':

        rank_by = 'ks_max'
        rank_by_var = 'ks'
        rank_asc = False

        # df_results
        df_results = pd.DataFrame(results, columns=['chart', 'dimension', 'ks', 'p', 'score'])
        df_results['rank'] = df_results[rank_by_var].rank(method='first', ascending=rank_asc)
        df_results = df_results.sort_values('rank')

        # df_results_chart
        df_results_chart = df_results.groupby(['chart'])[['ks', 'p', 'score']].agg(['mean', 'min', 'max'])
        df_results_chart.columns = ['_'.join(col) for col in df_results_chart.columns]
        df_results_chart = df_results_chart.reset_index()
        df_results_chart = df_results_chart.rename(columns={"score_max": "score"})
        df_results_chart['rank'] = df_results_chart['score'].rank(method='first', ascending=rank_asc)

        df_results_chart = df_results_chart.sort_values('rank')

    else:

        raise ValueError(f'unknown model "{model["type"]}"')

    df_results_chart = df_results_chart.round(2)

    return df_results_chart

