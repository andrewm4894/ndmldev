from datetime import datetime

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
    r = requests.get(url)
    r_json = r.json()
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
    default_window_size = 60
    default_baseline_window_multiplier = 1
    now = int(datetime.now().timestamp())
    default_before = now
    default_after = now - default_window_size
    default_highlight_before = default_before
    default_highlight_after = default_after
    url = request.args.get('url', None)
    if url:
        url_parts = url.split(';')
        default_after = int([x.split('=')[1] for x in url_parts if x.startswith('after=')][0])
        default_before = int([x.split('=')[1] for x in url_parts if x.startswith('before=')][0])
        default_highlight_after = int([x.split('=')[1] for x in url_parts if x.startswith('highlight_after=')][0])
        default_highlight_before = int([x.split('=')[1] for x in url_parts if x.startswith('highlight_before=')][0])
    rank_by = request.args.get('rank_by', 'ks_mean')
    starts_with = request.args.get('rank_by', None)
    format = request.args.get('format', 'json')
    before = request.args.get('before', default_before)
    after = request.args.get('after', default_after)
    highlight_before = request.args.get('highlight_before', default_highlight_before)
    highlight_after = request.args.get('highlight_after', default_highlight_after)
    window_multiplier = request.args.get('window_multiplier', default_baseline_window_multiplier)
    window_size = highlight_before - highlight_after
    baseline_before = highlight_after - 1
    baseline_after = baseline_before - (window_size * window_multiplier)
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

