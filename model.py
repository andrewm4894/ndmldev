import logging
import numpy as np
import pandas as pd
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
from adtk.detector import (
    InterQuartileRangeAD, AutoregressionAD, GeneralizedESDTestAD, LevelShiftAD, PersistAD, QuantileAD, SeasonalAD,
    VolatilityShiftAD, MinClusterDetector, OutlierDetector, PcaAD, RegressionAD
)

from adtk.detector import InterQuartileRangeAD as ADTKDefault
from sklearn.cluster import KMeans, DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression


log = logging.getLogger(__name__)

pyod_models_supported = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]
adtk_models_supported = [
    'iqr', 'ar', 'esd', 'level', 'persist', 'quantile', 'seasonal', 'volatility', 'kmeans', 'dbscan', 'eliptic',
    'pcaad', 'linear'
]
adtk_models_lags_allowed = [
    'kmeans'
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
    if model['type'] in pyod_models_supported:
        results = do_pyod(model, colnames, arr_baseline, arr_highlight)
    elif model['type'] in ['mp', 'mp_approx']:
        results = do_mp(colnames, arr_baseline, arr_highlight, model=model['type'])
    elif model['type'] in adtk_models_supported:
        results = do_adtk(model, colnames, arr_baseline, arr_highlight)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)
    return results


def do_mp(colnames, arr_baseline, arr_highlight, model='mp'):
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
        if model == 'mp':
            mp = stumpy.stump(arr[:, n], m)[:, 0]
        elif model == 'mp_approx':
            approx = stumpy.scrump(arr[:, n], m, percentage=0.01, pre_scrump=True)
            for _ in range(20):
                approx.update()
            mp = approx.P_
        else:
            raise ValueError(f"... unknown model '{model}'")
        mp_highlight = mp[0:n_highlight]
        mp_thold = np.percentile(mp, 90)
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


def do_adtk(model, colnames, arr_baseline, arr_highlight):
    n_lags = model.get('n_lags', 0)
    model = model.get('type', 'iqr')
    df_baseline = pd.DataFrame(arr_baseline, columns=colnames)
    df_baseline = df_baseline.set_index(pd.DatetimeIndex(pd.to_datetime(df_baseline.index, unit='s'), freq='1s'))
    df_highlight = pd.DataFrame(arr_highlight, columns=colnames)
    df_highlight = df_highlight.set_index(pd.DatetimeIndex(pd.to_datetime(df_highlight.index, unit='s'), freq='1s'))
    results = {}
    # loop over each col and do the ks test
    for colname in df_baseline.columns:

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]

        log.info(f'... chart = {chart}')
        log.info(f'... dimension = {dimension}')

        # check for bad data
        bad_data = False
        baseline_dim_na_pct = df_baseline[colname].isna().sum() / len(df_baseline)
        highlight_dim_na_pct = df_highlight[colname].isna().sum() / len(df_highlight)
        if baseline_dim_na_pct >= 0.1:
            bad_data = True
        if highlight_dim_na_pct >= 0.1:
            bad_data = True

        # skip if bad data
        if bad_data:

            log.info(f'... skipping due to bad data')

        else:

            df_baseline_dim = df_baseline[[colname]]
            df_highlight_dim = df_highlight[[colname]]

            if model in adtk_models_lags_allowed:
                if n_lags > 0:
                    df_baseline_dim = pd.concat([df_baseline_dim.shift(n_lag) for n_lag in range(n_lags+1)], axis=1)
                    df_highlight_dim = pd.concat([df_highlight_dim.shift(n_lag) for n_lag in range(n_lags + 1)], axis=1)
                    colnames_updated = [colname] + [f'{colname}_lag{n_lag}' for n_lag in range(1, n_lags+1)]
                    df_baseline_dim.columns = colnames_updated
                    df_highlight_dim.columns = colnames_updated

            df_baseline_dim = df_baseline_dim.dropna()
            df_highlight_dim = df_highlight_dim.dropna()

            log.info(f'... chart = {chart}')
            log.info(f'... dimension = {dimension}')
            log.info(f'... df_baseline_dim.shape = {df_baseline_dim.shape}')
            log.info(f'... df_highlight_dim.shape = {df_highlight_dim.shape}')
            log.info(f'... df_baseline_dim = {df_baseline_dim}')
            log.info(f'... df_highlight_dim = {df_highlight_dim}')

            if model == 'iqr':
                clf = InterQuartileRangeAD()
            elif model == 'ar':
                clf = AutoregressionAD()
            elif model == 'esd':
                clf = GeneralizedESDTestAD()
            elif model == 'level':
                clf = LevelShiftAD(15)
            elif model == 'persist':
                clf = PersistAD(15)
            elif model == 'quantile':
                clf = QuantileAD()
            elif model == 'seasonal':
                clf = SeasonalAD()
            elif model == 'volatility':
                clf = VolatilityShiftAD(15)
            elif model == 'kmeans':
                kmeans = KMeans(n_clusters=2).fit(df_baseline_dim)
                clf = MinClusterDetector(kmeans)
            elif model == 'dbscan':
                clf = MinClusterDetector(DBSCAN)
            elif model == 'eliptic':
                clf = OutlierDetector(EllipticEnvelope)
            elif model == 'pcaad':
                clf = PcaAD()
            elif model == 'linear':
                clf = RegressionAD(LinearRegression, target=colname)
            else:
                clf = ADTKDefault()
            clf.fit(df_baseline_dim)
            #try:
            #    clf.fit(df_baseline[colname])
            #except Exception as e:
            #    log.warning(e)
            #    clf = ADTKDefault()
            #    clf.fit(df_baseline[colname])
            preds = clf.predict(df_highlight_dim)
            score = np.mean(preds)
            if chart in results:
                results[chart].append({dimension: {'score': score}})
            else:
                results[chart] = [{dimension: {'score': score}}]
    print(results)
    print(xxx)
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

