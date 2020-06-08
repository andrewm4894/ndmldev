import numpy as np
from scipy.stats import ks_2samp
from pyod.models.cblof import CBLOF as PyODModel
from pyod.models.hbos import HBOS as DefaultPyODModel

from data import add_lags


def do_ks(colnames, arr_baseline, arr_highlight):
    results = []
    for n in range(arr_baseline.shape[1]):
        ks_stat, p_value = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        results.append([ks_stat, p_value])
    results = zip([[col.split('__')[0], col.split('__')[1]] for col in colnames], results)
    results = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in results]
    return results


def do_pyod(chart_cols, arr_baseline, arr_highlight, n_lags=2):
    if n_lags > 0:
        arr_baseline = add_lags(arr_baseline, n_lags=n_lags)
        arr_highlight = add_lags(arr_highlight, n_lags=n_lags)
    results = []
    model = PyODModel(contamination=0.1, n_clusters=2)
    for chart in chart_cols:
        try:
            model.fit(arr_baseline[:, chart_cols[chart]])
        except:
            model = DefaultPyODModel(contamination=0.1)
            model.fit(arr_baseline[:, chart_cols[chart]])
        preds = model.predict(arr_highlight[:, chart_cols[chart]])
        probs = model.predict_proba(arr_highlight[:, chart_cols[chart]])[:, 1]
        results.append([chart, np.mean(probs), np.mean(preds)])
    return results

