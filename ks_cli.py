import argparse
import multiprocessing
import time
from multiprocessing import Pool
from threading import Thread
from urllib.parse import parse_qs, urlparse

import trio
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp

from get_data import get_charts_df_async, get_chart_df
from ks import do_ks, rank_results
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
    '--run_mode', type=str, nargs='?', default='default'
)
parser.add_argument(
    '--ks_mode', type=str, nargs='?', default='default'
)
args = parser.parse_args()

# parse args
url = args.url
remote = args.remote
run_mode = args.run_mode
ks_mode = args.ks_mode

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

if run_mode == 'async':

    api_calls = [
        (f'http://{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json', chart)
        for chart in charts
    ]
    df = trio.run(get_charts_df_async, api_calls)
    df = df._get_numeric_data()
    df = filter_useless_cols(df)
    time_got_data = time.time()
    print(f'... time start to data = {time_got_data - time_start}')

    if ks_mode == 'vec':

        data_baseline = df[(df.index >= baseline_after) & (df.index <= baseline_before)]._get_numeric_data().transpose().values
        data_highlight = df[(df.index >= highlight_after) & (df.index <= highlight_before)]._get_numeric_data().transpose().values

        ks_2samp_vec = np.vectorize(stats.ks_2samp, signature='(n),(m)->(),()')
        results_vec = ks_2samp_vec(data_baseline, data_highlight)
        results_vec = list(zip(results_vec[0], results_vec[1]))

    elif ks_mode == 'default':

        #df_baseline = df[(df.index >= baseline_after) & (df.index <= baseline_before)]
        #df_highlight = df[(df.index >= highlight_after) & (df.index <= highlight_before)]

        #df_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').copy()
        #df_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').copy()

        #df_baseline = df.iloc[baseline_after:baseline_before]
        #print(df_baseline.shape)
        #print(df_baseline.head())
        #xxx

        results = []
        for col in df.columns:
            results.append(
                ks_2samp(
                    df[col].sample(frac=0.5),
                    df[col].sample(frac=0.5)
                )
            )

        #for chart in charts:
        #    chart_cols = [col for col in df.columns if col.startswith(f'{chart}__')]
        #    ks_results = do_ks(df[chart_cols], baseline_after, baseline_before, highlight_after, highlight_before)
        #    if ks_results:
        #        results[chart] = ks_results

    time_got_ks = time.time()
    print(f'... time data to ks = {time_got_ks - time_got_data}')
    #print(results)
    XXX

elif run_mode == 'default':

    for chart in charts:
        df = get_chart_df(chart, after=baseline_after, before=highlight_before, host=host)
        df = filter_useless_cols(df)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results

elif run_mode == 'multi':

    def do_multi(params):
        chart, baseline_after, baseline_before, highlight_after, highlight_before = params
        df = get_chart_df(chart, after=baseline_after, before=highlight_before, host=host)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                return {chart: ks_results}
            else:
                return None
        else:
            return None

    p = Pool(processes=(multiprocessing.cpu_count()-1))
    params_list = [(chart, baseline_after, baseline_before, highlight_after, highlight_before) for chart in charts]
    results = p.map(do_multi, params_list)
    results = [result for result in results if result]
    results = {list(d)[0]: d[list(d)[0]] for d in results}

elif run_mode == 'thread':

    results = {}

    def do_thread(params):
        chart, baseline_after, baseline_before, highlight_after, highlight_before = params
        df = get_chart_df(chart, after=baseline_after, before=highlight_before, host=host)
        if len(df) > 0:
            ks_results = do_ks(df, baseline_after, baseline_before, highlight_after, highlight_before)
            if ks_results:
                results[chart] = ks_results

    threads = []
    params_list = [(chart, baseline_after, baseline_before, highlight_after, highlight_before) for chart in charts]
    for param in params_list:
        process = Thread(target=do_thread, args=[param])
        process.start()
        threads.append(process)

    for process in threads:
        process.join()

results = rank_results(results, rank_by, ascending=False)
#print(results)

time_done = time.time()
print(f'... time total = {time_done - time_start}')
