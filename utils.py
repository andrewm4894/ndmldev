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


def get_chart_list():
    url = "http://127.0.0.1:19999/api/v1/charts"
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_list = [chart for chart in charts]
    return chart_list


def get_chart_df(chart, after, before, host: str = '127.0.0.1:19999', format: str = 'json', numeric_only: bool = True):
    url = f"http://{host}/api/v1/data?chart={chart}&after={after}&before={before}&format={format}"
    r = requests.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    print(df.shape)
    print(df.head())
    if numeric_only:
        df = df._get_numeric_data()
    df = df.set_index('time_idx')
    print(df.shape)
    print(df.head())
    return df


def filter_useless_cols(df):
    s = (df.min() == df.max())
    useless_cols = list(s.where(s == True).dropna().index)
    df = df.drop(useless_cols, axis=1)
    return df


def do_ks(df, baseline_start, baseline_end, window_start, window_end):

    print(df.shape)
    print(df.head())

    df = df._get_numeric_data()
    df = filter_useless_cols(df)

    print(df.shape)
    print(df.head())

    if len(df.columns) > 0:

        df_baseline = df[(df.index >= baseline_start) & (df.index <= baseline_end)]
        print(df_baseline.shape)
        df_window = df[(df.index >= window_start) & (df.index <= window_end)]
        print(df_window.shape)

        results = {
            'summary': {},
            'detail': {}
        }

        for col in df_baseline.columns:

            res = ks_2samp(df_baseline[col], df_window[col])
            results['detail'][col] = {"ks": float(round(res[0], 4)), "p": float(round(res[1], 4))}

        results['summary']['ks_mean'] = float(round(np.mean([results['detail'][res]['ks'] for res in results['detail']]), 4))
        results['summary']['ks_min'] = float(round(np.min([results['detail'][res]['ks'] for res in results['detail']]), 4))
        results['summary']['ks_max'] = float(round(np.max([results['detail'][res]['ks'] for res in results['detail']]), 4))
        results['summary']['p_mean'] = float(round(np.mean([results['detail'][res]['p'] for res in results['detail']]), 4))
        results['summary']['p_min'] = float(round(np.min([results['detail'][res]['p'] for res in results['detail']]), 4))
        results['summary']['p_max'] = float(round(np.max([results['detail'][res]['p'] for res in results['detail']]), 4))

        return results

    else:

        return None


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
    }
    return params

