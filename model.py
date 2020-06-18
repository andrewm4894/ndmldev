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
import stumpy


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


def run_model(model, colnames, arr_baseline, arr_highlight):
    """Function to take in data and some config and decide what model to run.
    """
    if model['type'] in supported_pyod_models:
        results = do_pyod(model, colnames, arr_baseline, arr_highlight)
    elif model['type'] == 'mp':
        results = do_mp(colnames, arr_baseline, arr_highlight)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)
    return results


def do_mp(colnames, arr_baseline, arr_highlight):
    arr = np.concatenate((arr_baseline, arr_highlight))
    n_baseline = arr_baseline.shape[0]
    n_highlight = arr_highlight.shape[0]
    # dict to collect results into
    results = {}
    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        m = 30
        #mp = stumpy.stump(arr[:, n], m)[:, 0]
        approx = stumpy.scrump(arr[:, n], m, percentage=0.01, pre_scrump=False)[:, 0]
        for _ in range(9):
            approx.update()
        mp = approx.P_[:, 0]
        mp_baseline = mp[0:n_baseline]
        mp_highlight = mp[0:n_highlight]
        mp_thold = np.mean(mp)
        score = np.mean(np.where(mp_highlight >= mp_thold, 1, 0))
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]
    return results


def do_ks(colnames, arr_baseline, arr_highlight):
    # dict to collect results into
    results = {}
    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        score, _ = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]
    return results


def do_pyod(model, colnames, arr_baseline, arr_highlight):
    n_lags = model.get('n_lags', 0)
    # dict to collect results into
    results = {}
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
    # fit model for each dimension and then use model to score highlighted area
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        arr_baseline_dim = arr_baseline[:, [n]]
        arr_highlight_dim = arr_highlight[:, [n]]
        if n_lags > 0:
            arr_baseline_dim = add_lags(arr_baseline_dim, n_lags=n_lags)
            arr_highlight_dim = add_lags(arr_highlight_dim, n_lags=n_lags)
        # remove any nan rows
        arr_baseline_dim = arr_baseline_dim[~np.isnan(arr_baseline_dim).any(axis=1)]
        arr_highlight_dim = arr_highlight_dim[~np.isnan(arr_highlight_dim).any(axis=1)]
        #log.info(f'... chart = {chart}')
        #log.info(f'... dimension = {dimension}')
        #log.info(f'... arr_baseline_dim.shape = {arr_baseline_dim.shape}')
        #log.info(f'... arr_highlight_dim.shape = {arr_highlight_dim.shape}')
        #log.info(f'... arr_baseline_dim = {arr_baseline_dim}')
        #log.info(f'... arr_highlight_dim = {arr_highlight_dim}')
        # try fit and if fails fallback to default model
        clf.fit(arr_baseline_dim)
        #try:
        #    clf.fit(arr_baseline_dim)
        #except:
        #    clf = DefaultPyODModel()
        #    clf.fit(arr_baseline_dim)
        # 0/1 anomaly predictions
        preds = clf.predict(arr_highlight_dim)
        #log.info(f'... preds.shape = {preds.shape}')
        #log.info(f'... preds = {preds}')
        # anomaly probability scores
        probs = clf.predict_proba(arr_highlight_dim)[:, 1]
        #log.info(f'... probs.shape = {probs.shape}')
        #log.info(f'... probs = {probs}')
        # save results
        score = (np.mean(probs) + np.mean(preds))/2
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]
    return results

