import logging
import numpy as np
from scipy.stats import ks_2samp
from pyod.models.hbos import HBOS as DefaultPyODModel
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from pyod.models.pca import PCA

log = logging.getLogger('model')

supported_pyod_models = ['knn', 'hbos', 'cblof', 'abod', 'pca']


def add_lags(arr, n_lags=1):
    arr_orig = np.copy(arr)
    for n_lag in range(1, n_lags + 1):
        arr = np.concatenate((arr, np.roll(arr_orig, n_lag, axis=0)), axis=1)
    arr = arr[n_lags:]
    return arr


def run_model(model, charts, colnames, arr_baseline, arr_highlight):
    """Function to take in data and some config and decide what model to run.
    """
    if model['type'] in supported_pyod_models:
        results = do_pyod(model, charts, colnames, arr_baseline, arr_highlight)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)
    return results


def do_ks(colnames, arr_baseline, arr_highlight):
    # list to collect results into
    results = []
    # loop over each col and do the ks test
    for n in range(arr_baseline.shape[1]):
        ks_stat, p_value = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        results.append([ks_stat, p_value])
    # wrangle results
    results = zip([[col.split('__')[0], col.split('__')[1]] for col in colnames], results)
    # ('chart', 'dimension', 'ks', 'p')
    results = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in results]
    return results


def do_pyod(model, charts, colnames, arr_baseline, arr_highlight):
    # list to collect results into
    results = []
    # map cols from array to charts
    chart_cols = {}
    for chart in charts:
        chart_cols[chart] = [colnames.index(col) for col in colnames if col.startswith(chart)]
    log.info(f'... chart_cols = {chart_cols}')
    # add lags if specified
    n_lags = model.get('n_lags', 0)
    if n_lags > 0:
        arr_baseline = add_lags(arr_baseline, n_lags=n_lags)
        arr_highlight = add_lags(arr_highlight, n_lags=n_lags)
    # initial model set up
    if model['type'] == 'knn':
        clf = KNN(**model['params'])
    elif model['type'] == 'cblof':
        clf = CBLOF(**model['params'])
    elif model['type'] == 'abod':
        clf = ABOD(**model['params'])
    elif model['type'] == 'pca':
        clf = PCA(**model['params'])
    else:
        clf = HBOS(**model['params'])
    # fit model for each chart and then use model to score highlighted area
    for chart in chart_cols:
        arr_baseline_chart = arr_baseline[:, chart_cols[chart]]
        arr_highlight_chart = arr_highlight[:, chart_cols[chart]]
        log.info(f'... chart = {chart}')
        log.info(f'... arr_baseline_chart.shape = {arr_baseline_chart.shape}')
        log.info(f'... arr_highlight_chart.shape = {arr_highlight_chart.shape}')
        log.info(f'... arr_baseline_chart = {arr_baseline_chart}')
        log.info(f'... arr_highlight_chart = {arr_highlight_chart}')
        # try fit and if fails fallback to default model
        try:
            clf.fit(arr_baseline_chart)
        except:
            clf = DefaultPyODModel()
            clf.fit(arr_baseline_chart)
        # 0/1 anomaly predictions
        preds = clf.predict(arr_highlight_chart)
        # anomaly probability scores
        probs = clf.predict_proba(arr_highlight_chart)[:, 1]
        # save results
        results.append([chart, np.mean(probs), np.mean(preds)])
    return results

