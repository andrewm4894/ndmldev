import asks
import requests

import pandas as pd
import trio

from ks import rank_results
from utils import get_chart_list


def get_chart_df(chart, after, before, host: str = '127.0.0.1:19999', format: str = 'json', numeric_only: bool = True):
    url = f"http://{host}/api/v1/data?chart={chart}&after={after}&before={before}&format={format}"
    #print(url)
    r = requests.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    #print(df.shape)
    if len(df) == 0:
        return df
    else:
        if numeric_only:
            df = df._get_numeric_data()
        df = df.set_index('time_idx')
        return df


async def get_chart_df_async(api_call, data):
    url, chart = api_call
    #print(url)
    r = await asks.get(url)
    r_json = r.json()
    df = pd.DataFrame(r_json['data'], columns=['time_idx'] + r_json['labels'][1:])
    df = df.set_index('time_idx').add_prefix(f'{chart}__')
    #print(df.shape)
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

