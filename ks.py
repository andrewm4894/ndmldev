import pandas as pd
from scipy.stats import ks_2samp


def do_ks(colnames, arr_baseline, arr_highlight):
    results = []
    for n in range(arr_baseline.shape[1]):
        ks_stat, p_value = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        results.append([ks_stat, p_value])
    results = zip([[col.split('__')[0], col.split('__')[1]] for col in colnames], results)
    results = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in results]
    return results


def results_to_df(results, rank_by, rank_asc):

    rank_by_var = rank_by.split('_')[0]

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

    return df_results, df_results_chart
