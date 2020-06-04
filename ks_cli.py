import argparse
import time
from urllib.parse import parse_qs, urlparse

import trio
import pandas as pd
from scipy.stats import ks_2samp

from get_data import get_charts_df_async
from utils import get_chart_list, filter_useless_cols

time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--url', type=str, nargs='?', help='url', default='http://127.0.0.1:19999/'
)
parser.add_argument(
    '--remote', type=str, nargs='?', default='no'
)
parser.add_argument(
    '--rank_by', type=str, nargs='?', default='ks_max'
)
parser.add_argument(
    '--rank_asc', type=bool, nargs='?', default=False
)
args = parser.parse_args()

# parse args
url = args.url
remote = args.remote
rank_by = args.rank_by
rank_asc = args.rank_asc

rank_by_var = rank_by.split('_')[0]
rank_by_agg = rank_by.split('_')[1]

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

results = {}
charts = get_chart_list(starts_with=starts_with, host=host)

# get data
api_calls = [
    (f'http://{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json', chart)
    for chart in charts
]
df = trio.run(get_charts_df_async, api_calls)
df = df._get_numeric_data()
df = filter_useless_cols(df)
arr_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').values
arr_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').values
time_got_data = time.time()
print(f'... time start to data = {time_got_data - time_start}')

# get ks
results = []
for n in range(arr_baseline.shape[1]):
    ks_stat, p_value = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
    results.append([ks_stat, p_value])
time_got_ks = time.time()
print(f'... time data to ks = {round(time_got_ks - time_got_data,2)}')

# wrangle results
results = zip([[col.split('__')[0], col.split('__')[1]] for col in list(df.columns)], results)
results = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in results]

# df_results
df_results = pd.DataFrame(results, columns=['chart', 'dimension', 'ks', 'p'])
df_results['rank'] = df_results[rank_by_var].rank(method='first', ascending=rank_asc)
df_results = df_results.sort_values('rank')

# df_results_chart
df_results_chart = df_results.groupby(['chart'])[['ks', 'p']].agg(['mean', 'min', 'max'])
df_results_chart.columns = ['_'.join(col) for col in df_results_chart.columns]
df_results_chart = df_results_chart.reset_index()
df_results_chart['rank'] = df_results_chart[rank_by].rank(method='first', ascending=rank_asc)
df_results_chart = df_results_chart.sort_values('rank')

time_got_results = time.time()
print(f'... time ks to results = {round(time_got_results - time_got_ks,2)}')

time_done = time.time()
print(f'... time total = {round(time_done - time_start,2)}')

print(df_results)
print(df_results_chart)

