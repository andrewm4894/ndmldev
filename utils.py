import json
import logging
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pandas as pd
import numpy as np

from model import chart_level_models

log = logging.getLogger(__name__)


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
            "n_lags": 0,
            "model_level": "dim"
        },
        "return_type": "html",
        "baseline_window_multiplier": 2,
        "score_thold": 0.2
    }

    pyod_config_default = {
        "model": {
            "type": "hbos",
            "params": {"contamination": 0.1},
            "n_lags": 2,
            "model_level": "dim"
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


def results_to_df(results_dict, score_thold):

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
                score_norm = (score - score_min) / (score_max - score_min)
                results_list.append([chart, k, score, score_norm])

    df_results = pd.DataFrame(results_list, columns=['chart', 'dimension', 'score', 'score_norm'])
    if score_thold > 0:
        df_results = df_results[df_results['score_norm'] >= score_thold]
    df_results['rank'] = df_results['score'].rank(method='first', ascending=False)
    df_results['chart_rank'] = df_results['chart'].map(
        df_results.groupby('chart')[['score']].mean().rank(
            method='first', ascending=False
        )['score'].to_dict()
    )
    df_results = df_results.sort_values('chart_rank', ascending=True)

    return df_results


def add_lags(data, n_lags=1, data_type='np'):
    data_orig = np.copy(data)
    if data_type == 'np':
        for n_lag in range(1, n_lags + 1):
            data = np.concatenate((data, np.roll(data_orig, n_lag, axis=0)), axis=1)
        data = data[n_lags:]
    elif data_type == 'df':
        colnames_new = [f"{col}_lag{n_lag}".replace('_lag0', '') for n_lag in range(n_lags+1) for col in data.columns]
        data = pd.concat([data.shift(n_lag) for n_lag in range(n_lags + 1)], axis=1)
        data.columns = colnames_new
    log.debug(f'... (add_lags) n_lags = {n_lags} arr_orig.shape = {data_orig.shape}  arr.shape = {data.shape}')
    return data


