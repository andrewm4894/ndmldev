import logging

import asks

import pandas as pd
import trio
from utils import filter_useless_cols, filter_lowstd_cols

log = logging.getLogger('data')


async def get_chart_df_async(api_call, data):
    url, chart = api_call
    r = await asks.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    df = df.set_index('time_idx').add_prefix(f'{chart}|')
    data[chart] = df


async def get_charts_df_async(api_calls):
    data = {}
    with trio.move_on_after(60):
        async with trio.open_nursery() as nursery:
            for api_call in api_calls:
                nursery.start_soon(get_chart_df_async, api_call, data)
    df = pd.concat(data, join='outer', axis=1, sort=True)
    df.columns = df.columns.droplevel()
    return df


def get_data(host, charts, baseline_after, baseline_before, highlight_after, highlight_before):
    api_calls = [
        (
            f'http://{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json',
            chart
        )
        for chart in charts
    ]
    df = trio.run(get_charts_df_async, api_calls)
    log.info(f'... df.shape = {df.shape}')
    df = df._get_numeric_data()
    log.info(f'... df.shape = {df.shape} (drop nonnumeric)')
    df = df.diff().dropna(how='all')
    log.info(f'... df.shape = {df.shape} (diff & drop na all)')
    df = filter_useless_cols(df)
    log.info(f'... df.shape = {df.shape} (filter useless)')
    df = filter_lowstd_cols(df)
    log.info(f'... df.shape = {df.shape} (filter lowstd)')
    log.info(f"... df.describe = {df.describe(include='all').transpose()}")
    colnames = list(df.columns)
    log.info(f'... colnames = {colnames}')
    arr_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').values
    arr_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').values
    return colnames, arr_baseline, arr_highlight





