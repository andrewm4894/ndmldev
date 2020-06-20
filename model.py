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
from adtk.detector import InterQuartileRangeAD as ADTKDefault


log = logging.getLogger(__name__)

pyod_models_supported = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]
adtk_models_supported = [
    'iqr', 'ar', 'esd', 'level', 'persist', 'quantile', 'seasonal', 'volatility', 'kmeans', 'birch', 'eliptic',
    'pcaad', 'linear', 'gmm', 'vbgmm', 'isof', 'lofad', 'mcdad', 'rf'
]
adtk_models_lags_allowed = [
    'kmeans', 'birch', 'gmm', 'eliptic', 'vbgmm', 'isof', 'lofad', 'mcdad', 'linear', 'rf'
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

    clf = adtk_init(model)
    n_dims = len(colnames)
    n_bad_data = 0
    fit_success = 0
    fit_fail = 0
    fit_default = 0

    results = {}

    # loop over each col and do the ks test
    for colname in df_baseline.columns:

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]

        #log.info(f'... chart = {chart}')
        #log.info(f'... dimension = {dimension}')

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

            n_bad_data += 1
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

            #log.info(f'... chart = {chart}')
            #log.info(f'... dimension = {dimension}')
            #log.info(f'... df_baseline_dim.shape = {df_baseline_dim.shape}')
            #log.info(f'... df_highlight_dim.shape = {df_highlight_dim.shape}')
            #log.info(f'... df_baseline_dim = {df_baseline_dim}')
            #log.info(f'... df_highlight_dim = {df_highlight_dim}')

            if model == 'linear':
                from adtk.detector import RegressionAD
                from sklearn.linear_model import LinearRegression
                clf = RegressionAD(LinearRegression(), target=colname)
            elif model == 'rf':
                from adtk.detector import RegressionAD
                from sklearn.ensemble import RandomForestRegressor
                clf = RegressionAD(RandomForestRegressor(), target=colname)

            try:
                clf.fit(df_baseline_dim)
                fit_success += 1
            except Exception as e:
                fit_fail += 1
                log.warning(e)
                log.info(f'... could not fit model for {colname}, trying default')
                clf = ADTKDefault()
                clf.fit(df_baseline_dim)
                fit_default += 1

            # get scores
            preds = clf.predict(df_highlight_dim)
            score = preds.mean().mean()
            if chart in results:
                results[chart].append({dimension: {'score': score}})
            else:
                results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    bad_data_rate = round(n_bad_data / n_dims, 2)
    success_rate = round(fit_success / n_dims, 2)
    log.info(f'... success_rate={success_rate}, bad_data_rate={bad_data_rate}, dims={n_dims}, bad_data={n_bad_data}, fit_success={fit_success}, fit_fail={fit_fail}, fit_default={fit_default}')

    return results


def do_pyod(model, colnames, arr_baseline, arr_highlight):

    n_lags = model.get('n_lags', 0)

    # dict to collect results into
    results = {}

    # initial model set up
    clf = pyod_init(model)

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


def pyod_init(model):
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
    return clf


def adtk_init(model):
    if model == 'iqr':
        from adtk.detector import InterQuartileRangeAD
        clf = InterQuartileRangeAD()
    elif model == 'ar':
        from adtk.detector import AutoregressionAD
        clf = AutoregressionAD()
    elif model == 'esd':
        from adtk.detector import GeneralizedESDTestAD
        clf = GeneralizedESDTestAD()
    elif model == 'level':
        from adtk.detector import LevelShiftAD
        clf = LevelShiftAD(15)
    elif model == 'persist':
        from adtk.detector import PersistAD
        clf = PersistAD(15)
    elif model == 'quantile':
        from adtk.detector import QuantileAD
        clf = QuantileAD()
    elif model == 'seasonal':
        from adtk.detector import SeasonalAD
        clf = SeasonalAD()
    elif model == 'volatility':
        from adtk.detector import VolatilityShiftAD
        clf = VolatilityShiftAD(15)
    elif model == 'kmeans':
        from adtk.detector import MinClusterDetector
        from sklearn.cluster import KMeans
        clf = MinClusterDetector(KMeans(n_clusters=2))
    elif model == 'birch':
        from adtk.detector import MinClusterDetector
        from sklearn.cluster import Birch
        clf = MinClusterDetector(Birch(threshold=0.25, branching_factor=25))
    elif model == 'gmm':
        from adtk.detector import MinClusterDetector
        from sklearn.mixture import GaussianMixture
        clf = MinClusterDetector(GaussianMixture(n_components=2, max_iter=50))
    elif model == 'vbgmm':
        from adtk.detector import MinClusterDetector
        from sklearn.mixture import BayesianGaussianMixture
        clf = MinClusterDetector(BayesianGaussianMixture(n_components=2, max_iter=50))
    elif model == 'eliptic':
        from adtk.detector import OutlierDetector
        from sklearn.covariance import EllipticEnvelope
        clf = OutlierDetector(EllipticEnvelope())
    elif model == 'mcdad':
        from adtk.detector import OutlierDetector
        from sklearn.covariance import MinCovDet
        clf = OutlierDetector(MinCovDet())
    elif model == 'isof':
        from adtk.detector import OutlierDetector
        from sklearn.ensemble import IsolationForest
        clf = OutlierDetector(IsolationForest())
    elif model == 'lofad':
        from adtk.detector import OutlierDetector
        from sklearn.neighbors import LocalOutlierFactor
        clf = OutlierDetector(LocalOutlierFactor())
    elif model == 'pcaad':
        from adtk.detector import PcaAD
        clf = PcaAD()
    #elif model == 'linear':
    #    from adtk.detector import RegressionAD
    #    from sklearn.linear_model import LinearRegression
    #    clf = RegressionAD(LinearRegression())
    else:
        clf = ADTKDefault()
    return clf

