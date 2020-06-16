import logging
import numpy as np
from scipy.stats import ks_2samp
from pyod.models.pca import PCA as DefaultPyODModel
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.vae import VAE
from pyod.models.xgbod import XGBOD


log = logging.getLogger(__name__)

supported_pyod_models = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]


def add_lags(arr, n_lags=1):
    arr_orig = np.copy(arr)
    for n_lag in range(1, n_lags + 1):
        arr = np.concatenate((arr, np.roll(arr_orig, n_lag, axis=0)), axis=1)
    arr = arr[n_lags:]
    #log.info(f'... (add_lags) n_lags = {n_lags} arr_orig.shape = {arr_orig.shape}  arr.shape = {arr.shape}')
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
    # get max and min to normalize ks score
    ks_max = max([result[0] for result in results])
    ks_min = min([result[0] for result in results])
    # wrangle results
    results = zip([[col.split('|')[0], col.split('|')[1]] for col in colnames], results)
    # ('chart', 'dimension', 'ks', 'p', 'score')
    results = [[x[0][0], x[0][1], x[1][0], x[1][1], (x[1][0]-ks_min)/(ks_max-ks_min)] for x in results]
    return results


def do_pyod(model, charts, colnames, arr_baseline, arr_highlight):
    n_lags = model.get('n_lags', 0)
    model_level = model.get('model_level', 'dimension')
    # list to collect results into
    results = []
    # map cols from array to charts
    chart_cols = {}
    if model_level == 'chart':
        for chart in charts:
            chart_cols[chart] = [colnames.index(col) for col in colnames if col.startswith(chart)]
    elif model_level == 'dimension':
        for col in colnames:
            chart_cols[col] = [colnames.index(col) for col in colnames]
    else:
        raise ValueError(f'invalid model_level {model_level}')
    log.info(f'... chart_cols = {chart_cols}')
    # initial model set up
    if model['type'] == 'knn':
        clf = KNN(**model['params'])
    elif model['type'] == 'abod':
        clf = ABOD(**model['params'])
    elif model['type'] == 'auto_encoder':
        clf = AutoEncoder(**model['params'])
    elif model['type'] == 'cblof':
        clf = CBLOF(**model['params'])
    elif model['type'] == 'hbos':
        clf = HBOS(**model['params'])
    elif model['type'] == 'iforest':
        clf = IForest(**model['params'])
    elif model['type'] == 'lmdd':
        clf = LMDD(**model['params'])
    elif model['type'] == 'loci':
        clf = LOCI(**model['params'])
    elif model['type'] == 'loda':
        clf = LODA(**model['params'])
    elif model['type'] == 'lof':
        clf = LOF(**model['params'])
    elif model['type'] == 'mcd':
        clf = MCD(**model['params'])
    elif model['type'] == 'ocsvm':
        clf = OCSVM(**model['params'])
    elif model['type'] == 'pca':
        clf = PCA(**model['params'])
    elif model['type'] == 'sod':
        clf = SOD(**model['params'])
    elif model['type'] == 'vae':
        clf = VAE(**model['params'])
    elif model['type'] == 'xgbod':
        clf = XGBOD(**model['params'])
    else:
        clf = DefaultPyODModel(**model['params'])
    # fit model for each chart and then use model to score highlighted area
    for chart in chart_cols:
        arr_baseline_chart = arr_baseline[:, chart_cols[chart]]
        arr_highlight_chart = arr_highlight[:, chart_cols[chart]]
        if n_lags > 0:
            arr_baseline_chart = add_lags(arr_baseline_chart, n_lags=n_lags)
            arr_highlight_chart = add_lags(arr_highlight_chart, n_lags=n_lags)
        #log.info(f'... chart = {chart}')
        #log.info(f'... arr_baseline_chart.shape = {arr_baseline_chart.shape}')
        #log.info(f'... arr_highlight_chart.shape = {arr_highlight_chart.shape}')
        #log.info(f'... arr_baseline_chart = {arr_baseline_chart}')
        #log.info(f'... arr_highlight_chart = {arr_highlight_chart}')
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

