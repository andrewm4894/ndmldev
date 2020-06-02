from collections import OrderedDict

from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

from utils import filter_useless_cols


def do_ks(df, baseline_start, baseline_end, highlight_start, highlight_end):

    df = df._get_numeric_data()
    df = filter_useless_cols(df)

    if len(df.columns) > 0:

        df_baseline = df[(df.index >= baseline_start) & (df.index <= baseline_end)]
        df_highlight = df[(df.index >= highlight_start) & (df.index <= highlight_end)]
        results = {
            'summary': {},
            'detail': {}
        }

        for col in df_baseline.columns:

            res = ks_2samp(df_baseline[col], df_highlight[col])
            results['detail'][col] = {"ks": float(round(res[0], 4)), "p": float(round(res[1], 4))}

        results['summary']['ks_mean'] = float(round(np.mean([results['detail'][res]['ks'] for res in results['detail']]), 4))
        results['summary']['ks_min'] = float(round(np.min([results['detail'][res]['ks'] for res in results['detail']]), 4))
        results['summary']['ks_max'] = float(round(np.max([results['detail'][res]['ks'] for res in results['detail']]), 4))
        results['summary']['p_mean'] = float(round(np.mean([results['detail'][res]['p'] for res in results['detail']]), 4))
        results['summary']['p_min'] = float(round(np.min([results['detail'][res]['p'] for res in results['detail']]), 4))
        results['summary']['p_max'] = float(round(np.max([results['detail'][res]['p'] for res in results['detail']]), 4))

        return results

    else:

        return None


def rank_results(results, rank_by, ascending: bool = True):
    df_rank = pd.DataFrame(data=[[c, results[c]['summary'][rank_by]] for c in results], columns=['chart', 'score'])
    df_rank['rank'] = df_rank['score'].rank(method='first', ascending=ascending)
    for _, row in df_rank.iterrows():
        results[row['chart']]['rank'] = int(row['rank'])
        results[row['chart']]['score'] = float(row['score'])
    results = OrderedDict(sorted(results.items(), key=lambda t: t[1]["rank"]))
    return results
