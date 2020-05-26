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
    df = pd.DataFrame(r_json['data'], columns=r_json['labels'])
    df = df.rename(columns={df.columns[1]: "time_idx"})
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
            results['detail'][col] = {"ks": round(res[0], 4), "p": round(res[1], 4)}

        results['summary']['ks_mean'] = round(np.mean([results['detail'][res]['ks'] for res in results['detail']]), 4)
        results['summary']['ks_min'] = round(np.min([results['detail'][res]['ks'] for res in results['detail']]), 4)
        results['summary']['ks_max'] = round(np.max([results['detail'][res]['ks'] for res in results['detail']]), 4)
        results['summary']['p_mean'] = round(np.mean([results['detail'][res]['p'] for res in results['detail']]), 4)
        results['summary']['p_min'] = round(np.min([results['detail'][res]['p'] for res in results['detail']]), 4)
        results['summary']['p_max'] = round(np.max([results['detail'][res]['p'] for res in results['detail']]), 4)

        return results

    else:

        return None

