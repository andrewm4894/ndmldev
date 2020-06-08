import argparse
import time
from urllib.parse import parse_qs, urlparse

from data import get_data
from ks import do_ks
from utils import get_chart_list, results_to_df

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

# get charts
charts = get_chart_list(starts_with=starts_with, host=host)

# get data
colnames, arr_baseline, arr_highlight = get_data(host, charts, baseline_after, baseline_before, highlight_after, highlight_before)
time_got_data = time.time()
print(f'... time start to data = {time_got_data - time_start}')

# do ks
results = do_ks(colnames, arr_baseline, arr_highlight)
time_got_ks = time.time()
print(f'... time data to ks = {round(time_got_ks - time_got_data,2)}')

# df_results
df_results, df_results_chart = results_to_df(results, rank_by, rank_asc)
time_got_results = time.time()
print(f'... time ks to results = {round(time_got_results - time_got_ks,2)}')

time_done = time.time()
print(f'... time total = {round(time_done - time_start,2)}')

print(df_results)
print(df_results_chart)

