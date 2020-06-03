import asks
import requests

import pandas as pd
import trio

from ks import rank_results, do_ks
from utils import get_chart_list


def get_chart_df(chart, after, before, host: str = '127.0.0.1:19999', format: str = 'json', numeric_only: bool = True):
    url = f"http://{host}/api/v1/data?chart={chart}&after={after}&before={before}&format={format}"
    r = requests.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    if len(df) == 0:
        return df
    else:
        if numeric_only:
            df = df._get_numeric_data()
        df = df.set_index('time_idx')
        return df


async def get_chart_df_async(api_call, data):
    url, chart = api_call
    r = await asks.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    df = df.set_index('time_idx').add_prefix(f'{chart}__')
    data[chart] = df


async def get_charts_df_async(api_calls):
    data = {}
    with trio.move_on_after(5):
        async with trio.open_nursery() as nursery:
            for api_call in api_calls:
                nursery.start_soon(get_chart_df_async, api_call, data)
    df = pd.concat(data, join='outer', axis=1, sort=True)
    df.columns = df.columns.droplevel()
    return df


async def do_it_all(starts_with, host, rank_by, baseline_after, baseline_before, highlight_after, highlight_before):
    results = {}
    charts = get_chart_list(starts_with=starts_with, host=host)
    api_calls = [
        (f'http://{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json', chart)
        for chart in charts
    ]
    data = {}
    with trio.move_on_after(5):
        async with trio.open_nursery() as nursery:
            for api_call in api_calls:
                nursery.start_soon(get_chart_df_async, api_call, data)
    df = pd.concat(data, join='outer', axis=1, sort=True)
    df.columns = df.columns.droplevel()
    for chart in charts:
        chart_cols = [col for col in df.columns if col.startswith(f'{chart}__')]
        df_chart = df[chart_cols]
        ks_results = do_ks(df_chart, baseline_after, baseline_before, highlight_after, highlight_before)
        if ks_results:
            results[chart] = ks_results
    results = rank_results(results, rank_by, ascending=False)
    print(results)
    return df

