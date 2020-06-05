from scipy.stats import ks_2samp


def do_ks(colnames, arr_baseline, arr_highlight):
    results = []
    for n in range(arr_baseline.shape[1]):
        ks_stat, p_value = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        results.append([ks_stat, p_value])
    results = zip([[col.split('__')[0], col.split('__')[1]] for col in colnames], results)
    results = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in results]
    return results

