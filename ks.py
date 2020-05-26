from scipy.stats import ks_2samp

from utils import filter_useless_cols


def do_ks(df, baseline_start, baseline_end, window_start, window_end):
    df = df._get_numeric_data()
    df = filter_useless_cols(df)
    if len(df.columns) > 0:
        df_baseline = df[(df.index >= baseline_start) & (df.index <= baseline_end)]
        print(df_baseline.shape)
        df_window = df[(df.index >= window_start) & (df.index <= window_end)]
        print(df_window.shape)
        results = {
            'summary': {},
            'detail': {}
        }
        for col in df_baseline.columns:
            res = ks_2samp(df_baseline[col], df_window[col])
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

