import argparse
import json
import time
from urllib.parse import parse_qs, urlparse

from data import get_data

from model import run_model
from utils import get_chart_list, results_to_df

time_start = time.time()

config_default ="""
    {
        "model": {
            "type": "ks",
            "params": {},
            "n_lags": 0
        },
        "return_type": "html",
        "baseline_window_multiplier": 2
    }
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    '--url', type=str, nargs='?', help='url', default='http://127.0.0.1:19999/'
)
parser.add_argument(
    '--remote', type=str, nargs='?', default='no'
)
parser.add_argument(
    '--config', type=str, nargs='?', default=config_default
)
parser.add_argument(
    '--rank_asc', type=bool, nargs='?', default=False
)
args = parser.parse_args()

# parse args
url = args.url
remote = args.remote
config = json.loads(args.config)
model = config['model']

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

# get scores
results = run_model(model, charts, colnames, arr_baseline, arr_highlight)
time_got_scores = time.time()
print(f'... time data to scores = {round(time_got_scores - time_got_data, 2)}')

# df_results_chart
df_results_chart = results_to_df(results, model)
time_got_results = time.time()
print(f'... time scores to results = {round(time_got_results - time_got_scores, 2)}')

time_done = time.time()
print(f'... time total = {round(time_done - time_start,2)}')

print(df_results_chart)

