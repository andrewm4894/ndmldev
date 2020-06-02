from datetime import datetime
import logging
from urllib.parse import parse_qs

import requests
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def get_chart_data_urls():
    url = "http://127.0.0.1:19999/api/v1/charts"
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_data_urls = {chart: charts[chart].get('data_url') for chart in charts}
    return chart_data_urls


def get_chart_list(starts_with: str = None):
    url = "http://127.0.0.1:19999/api/v1/charts"
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_list = [chart for chart in charts]
    if starts_with:
        chart_list = [chart for chart in chart_list if chart.startswith(starts_with)]
    return chart_list


def get_chart_df(chart, after, before, host: str = '127.0.0.1:19999', format: str = 'json', numeric_only: bool = True):
    url = f"http://{host}/api/v1/data?chart={chart}&after={after}&before={before}&format={format}"
    print(url)
    r = requests.get(url)
    r_json = r.json()
    print(r_json)
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    if numeric_only:
        df = df._get_numeric_data()
    df = df.set_index('time_idx')
    return df


def filter_useless_cols(df):
    s = (df.min() == df.max())
    useless_cols = list(s.where(s == True).dropna().index)
    df = df.drop(useless_cols, axis=1)
    return df


def parse_params(request):

    now = int(datetime.now().timestamp())
    default_window_size = 60 * 2
    baseline_window_multiplier = 2

    url_params = parse_qs(request.args.get('url'))
    if 'after' in url_params:
        after = int(url_params.get('after')[0]) / 1000
    else:
        after = int(request.args.get('after', now - default_window_size))
    if 'before' in url_params:
        before = int(url_params.get('before')[0]) / 1000
    else:
        before = int(request.args.get('before', now))
    if 'highlight_after' in url_params:
        highlight_after = int(url_params.get('highlight_after')[0]) / 1000
    else:
        highlight_after = request.args.get('highlight_after', after)
    if 'highlight_before' in url_params:
        highlight_before = int(url_params.get('highlight_before')[0]) / 1000
    else:
        highlight_before = request.args.get('highlight_before', before)

    rank_by = request.args.get('rank_by', 'ks_mean')
    starts_with = request.args.get('rank_by', 'system.')
    format = request.args.get('format', 'json')

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
        "rank_by": rank_by,
        "starts_with": starts_with,
        "format": format
    }
    return params

