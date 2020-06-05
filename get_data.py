import asks

import pandas as pd
import trio
from utils import filter_useless_cols


async def get_chart_df_async(api_call, data):
    url, chart = api_call
    r = await asks.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    df = df.set_index('time_idx').add_prefix(f'{chart}__')
    data[chart] = df


async def get_charts_df_async(api_calls):
    data = {}
    with trio.move_on_after(60):
        async with trio.open_nursery() as nursery:
            for api_call in api_calls:
                nursery.start_soon(get_chart_df_async, api_call, data)
    df = pd.concat(data, join='outer', axis=1, sort=True)
    df.columns = df.columns.droplevel()
    df = df._get_numeric_data()
    df = filter_useless_cols(df)
    df = df.diff().dropna(how='all')
    df = df._get_numeric_data()
    df = filter_useless_cols(df)
    return df


def get_data(host, charts, baseline_after, baseline_before, highlight_after, highlight_before):
    api_calls = [
        (
            "http://" + f'{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json'.replace('///', '/').replace('//', '/'),
            chart
        )
        for chart in charts
    ]
    df = trio.run(get_charts_df_async, api_calls)
    df = df._get_numeric_data()
    df = filter_useless_cols(df)
    colnames = list(df.columns)
    arr_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').values
    arr_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').values
    return colnames, arr_baseline, arr_highlight


