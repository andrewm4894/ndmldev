import argparse
import time
from urllib.parse import parse_qs, urlparse

import trio

from get_data import get_charts_df_async, get_chart_df
from ks import do_ks, rank_results
from utils import get_chart_list

time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--url', type=str, nargs='?', help='url', default='http://127.0.0.1:19999/'
)
parser.add_argument(
    '--remote', type=str, nargs='?', default='no'
)
parser.add_argument(
    '--run_mode', type=str, nargs='?', default='default'
)
args = parser.parse_args()

# parse args
url = args.url
remote = args.remote
run_mode = args.run_mode

baseline_window_multiplier = 2

url_params = parse_qs(url)
url_parse = urlparse(url)

if remote == "yes":
    host = url_parse.netloc
else:
    host = '127.0.0.1:19999'

after = int(int(url_params.get('after')[0]) / 1000)
before = int(int(url_params.get('before')[0]) / 1000)
highlight_after = int(int(url_params.get('highlight_after')[0]) / 1000)
highlight_before = int(int(url_params.get('highlight_before')[0]) / 1000)
starts_with = None
window_size = highlight_before - highlight_after
baseline_before = highlight_after - 1
baseline_after = baseline_before - (window_size * baseline_window_multiplier)
rank_by = 'ks_max'

results = {}
charts = get_chart_list(starts_with=starts_with, host=host)

if run_mode == 'aysnc':
    api_calls = [
        (f'http://{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json', chart)
        for chart in charts
    ]
    df = trio.run(get_charts_df_async, api_calls)
    for chart in charts:
        chart_cols = [col for col in df.columns if col.startswith(f'{chart}__')]
        df_chart = df[chart_cols]
        ks_results = do_ks(df_chart, baseline_after, baseline_before, highlight_after, highlight_before)
        if ks_results:
            results[chart] = ks_results
elif run_mode == 'default':
    for chart in charts:
        df = get_chart_df(chart, after=baseline_after, before=highlight_before, host=host)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results

results = rank_results(results, rank_by, ascending=False)

print(results)

time_done = time.time()
print('... total time = {}'.format(time_done - time_start))
