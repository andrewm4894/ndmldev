import logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pyod.models.pca import PCA as DefaultPyODModel
import stumpy
from adtk.detector import InterQuartileRangeAD as ADTKDefault


log = logging.getLogger(__name__)

pyod_models_supported = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]
adtk_models_supported = [
    'iqr', 'ar', 'esd', 'level', 'persist', 'quantile', 'seasonal', 'volatility', 'kmeans', 'birch', 'eliptic',
    'pcaad', 'linear', 'gmm', 'vbgmm', 'isof', 'lofad', 'mcdad', 'rf', 'huber', 'knnad', 'kernridge'
]
adtk_models_lags_allowed = [
    'kmeans', 'birch', 'gmm', 'eliptic', 'vbgmm', 'isof', 'lofad', 'mcdad', 'linear', 'rf', 'huber', 'knnad',
    'kernridge'
]
adtk_models_chart_level = [
    'kmeans', 'birch', 'gmm', 'eliptic', 'vbgmm', 'isof', 'lofad', 'mcdad', 'linear', 'rf', 'huber', 'knnad',
    'kernridge'
]
chart_level_models = pyod_models_supported + adtk_models_chart_level


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


def do_mp(colnames, arr_baseline, arr_highlight, model='mp', model_level='dim'):

    arr = np.concatenate((arr_baseline, arr_highlight))
    n_highlight = arr_highlight.shape[0]
    n_charts = len(set([colname.split('|')[0] for colname in colnames]))
    n_dims = len(colnames)
    n_bad_data = 0
    fit_success = 0
    fit_fail = 0
    fit_default = 0

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

        fit_success += 1
        mp_highlight = mp[0:n_highlight]
        mp_thold = np.percentile(mp, 90)

        score = np.mean(np.where(mp_highlight >= mp_thold, 1, 0))
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level))

    return results


def do_ks(colnames, arr_baseline, arr_highlight):

    # dict to collect results into
    results = {}
    n_charts = len(set([colname.split('|')[0] for colname in colnames]))
    n_dims = len(colnames)
    n_bad_data = 0
    fit_success = 0
    fit_fail = 0
    fit_default = 0

    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        score, _ = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        fit_success += 1
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default))

    return results


def do_adtk(model, colnames, arr_baseline, arr_highlight):

    n_lags = model.get('n_lags', 0)
    model_level = model.get('model_level', 'dim')
    model = model.get('type', 'iqr')

    df_baseline = pd.DataFrame(arr_baseline, columns=colnames)
    df_baseline = df_baseline.set_index(pd.DatetimeIndex(pd.to_datetime(df_baseline.index, unit='s'), freq='1s'))
    df_highlight = pd.DataFrame(arr_highlight, columns=colnames)
    df_highlight = df_highlight.set_index(pd.DatetimeIndex(pd.to_datetime(df_highlight.index, unit='s'), freq='1s'))

    clf = adtk_init(model)
    n_charts = len(set([colname.split('|')[0] for colname in colnames]))
    n_dims = len(colnames)
    n_bad_data = 0
    fit_success = 0
    fit_fail = 0
    fit_default = 0

    results = {}

    col_map = get_col_map(colnames, model, model_level)

    # build each model
    for colname in col_map:

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1] if '|' in colname else '*'

        log.debug(f'... chart = {chart}')
        log.debug(f'... dimension = {dimension}')

        df_baseline_dim = df_baseline.iloc[:, col_map[colname]]
        df_highlight_dim = df_highlight.iloc[:, col_map[colname]]

        # check for bad data
        bad_data = False
        baseline_dim_na_pct = max(df_baseline_dim.isna().sum() / len(df_baseline))
        highlight_dim_na_pct = max(df_highlight_dim.isna().sum() / len(df_highlight))
        if baseline_dim_na_pct >= 0.1:
            bad_data = True
        if highlight_dim_na_pct >= 0.1:
            bad_data = True

        # skip if bad data
        if bad_data:

            n_bad_data += 1
            log.info(f'... skipping {colname} due to bad data')

        else:

            if model in adtk_models_lags_allowed:

                if n_lags > 0:

                    df_baseline_dim = add_lags(df_baseline_dim, n_lags, 'df')
                    df_highlight_dim = add_lags(df_highlight_dim, n_lags, 'df')

            df_baseline_dim = df_baseline_dim.dropna()
            df_highlight_dim = df_highlight_dim.dropna()

            log.debug(f'... chart = {chart}')
            log.debug(f'... dimension = {dimension}')
            log.debug(f'... df_baseline_dim.shape = {df_baseline_dim.shape}')
            log.debug(f'... df_highlight_dim.shape = {df_highlight_dim.shape}')
            log.debug(f'... df_baseline_dim = {df_baseline_dim}')
            log.debug(f'... df_highlight_dim = {df_highlight_dim}')

            if model == 'linear':
                from adtk.detector import RegressionAD
                from sklearn.linear_model import LinearRegression
                clf = RegressionAD(LinearRegression(), target=colname)
            elif model == 'rf':
                from adtk.detector import RegressionAD
                from sklearn.ensemble import RandomForestRegressor
                clf = RegressionAD(RandomForestRegressor(), target=colname)
            elif model == 'huber':
                from adtk.detector import RegressionAD
                from sklearn.linear_model import HuberRegressor
                clf = RegressionAD(HuberRegressor(), target=colname)
            elif model == 'knnad':
                from adtk.detector import RegressionAD
                from sklearn.neighbors import KNeighborsRegressor
                clf = RegressionAD(KNeighborsRegressor(), target=colname)
            elif model == 'kernridge':
                from adtk.detector import RegressionAD
                from sklearn.kernel_ridge import KernelRidge
                clf = RegressionAD(KernelRidge(), target=colname)

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

            log.debug(f'... preds.shape = {preds.shape}')
            log.debug(f'... preds = {preds}')

            score = preds.mean().mean()
            if chart in results:
                results[chart].append({dimension: {'score': score}})
            else:
                results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level))

    return results


def do_pyod(model, colnames, arr_baseline, arr_highlight):

    n_lags = model.get('n_lags', 0)
    model_level = model.get('model_level', 'dim')
    model = model.get('type', 'hbos')

    # dict to collect results into
    results = {}

    # initial model set up
    clf = pyod_init(model)
    n_charts = len(set([colname.split('|')[0] for colname in colnames]))
    n_dims = len(colnames)
    n_bad_data = 0
    fit_success = 0
    fit_fail = 0
    fit_default = 0

    col_map = get_col_map(colnames, model, model_level)

    # build each model
    for colname in col_map:

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1] if '|' in colname else '*'
        arr_baseline_dim = arr_baseline[:, col_map[colname]]
        arr_highlight_dim = arr_highlight[:, col_map[colname]]

        # check for bad data
        bad_data = False

        # skip if bad data
        if bad_data:

            n_bad_data += 1
            log.info(f'... skipping {colname} due to bad data')

        else:

            if n_lags > 0:
                arr_baseline_dim = add_lags(arr_baseline_dim, n_lags=n_lags)
                arr_highlight_dim = add_lags(arr_highlight_dim, n_lags=n_lags)

            # remove any nan rows
            arr_baseline_dim = arr_baseline_dim[~np.isnan(arr_baseline_dim).any(axis=1)]
            arr_highlight_dim = arr_highlight_dim[~np.isnan(arr_highlight_dim).any(axis=1)]

            log.debug(f'... chart = {chart}')
            log.debug(f'... dimension = {dimension}')
            log.debug(f'... arr_baseline_dim.shape = {arr_baseline_dim.shape}')
            log.debug(f'... arr_highlight_dim.shape = {arr_highlight_dim.shape}')
            log.debug(f'... arr_baseline_dim = {arr_baseline_dim}')
            log.debug(f'... arr_highlight_dim = {arr_highlight_dim}')

            try:
                clf.fit(arr_baseline_dim)
                fit_success += 1
            except Exception as e:
                fit_fail += 1
                log.warning(e)
                log.info(f'... could not fit model for {colname}, trying default')
                clf = DefaultPyODModel()
                clf.fit(arr_baseline_dim)
                fit_default += 1

            # 0/1 anomaly predictions
            preds = clf.predict(arr_highlight_dim)

            log.debug(f'... preds.shape = {preds.shape}')
            log.debug(f'... preds = {preds}')

            # anomaly probability scores
            probs = clf.predict_proba(arr_highlight_dim)[:, 1]

            log.debug(f'... probs.shape = {probs.shape}')
            log.debug(f'... probs = {probs}')

            # save results
            score = (np.mean(probs) + np.mean(preds))/2
            if chart in results:
                results[chart].append({dimension: {'score': score}})
            else:
                results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level))

    return results


def get_col_map(colnames, model, model_level):
    col_map = {}
    if model_level == 'chart' and model in chart_level_models:
        charts_list = list(set([colname.split('|')[0] for colname in colnames]))
        for chart in charts_list:
            col_map[chart] = [colnames.index(colname) for colname in colnames if colname.startswith(f'{chart}|')]
    else:
        for col in colnames:
            col_map[col] = [colnames.index(colname) for colname in colnames if colname == col]
    return col_map


def summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level):
    # log some summary stats
    if model_level == 'chart':
        success_rate = round(fit_success / n_charts, 2)
        bad_data_rate = round(n_bad_data / n_charts, 2)
    else:
        bad_data_rate = round(n_bad_data / n_dims, 2)
        success_rate = round(fit_success / n_dims, 2)
    msg = f"... success_rate={success_rate}, bad_data_rate={bad_data_rate}, dims={n_dims}, bad_data={n_bad_data}"
    msg += f", fit_success={fit_success}, fit_fail={fit_fail}, fit_default={fit_default}'"
    return msg


def add_lags(data, n_lags=1, data_type='np'):
    data_orig = np.copy(data)
    if data_type == 'np':
        for n_lag in range(1, n_lags + 1):
            data = np.concatenate((data, np.roll(data_orig, n_lag, axis=0)), axis=1)
        data = data[n_lags:]
    elif data_type == 'df':
        colnames_new = [f"{col}_lag{n_lag}".replace('_lag0', '') for n_lag in range(n_lags) for col in data.columns]
        data = pd.concat([data.shift(n_lag) for n_lag in range(n_lags + 1)], axis=1)
        data.columns = colnames_new
    log.debug(f'... (add_lags) n_lags = {n_lags} arr_orig.shape = {data_orig.shape}  arr.shape = {data.shape}')
    return data


def pyod_init(model, n_features=None):
    # initial model set up
    if model == 'abod':
        from pyod.models.abod import ABOD
        clf = ABOD()
    elif model == 'auto_encoder':
        #import os
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from pyod.models.auto_encoder import AutoEncoder
        clf = AutoEncoder(
            hidden_neurons=[n_features, n_features*5, n_features*5, n_features], epochs=5,
            batch_size=64, preprocessing=False
        )
    elif model == 'cblof':
        from pyod.models.cblof import CBLOF
        clf = CBLOF(n_clusters=4)
    elif model == 'hbos':
        from pyod.models.hbos import HBOS
        clf = HBOS()
    elif model == 'iforest':
        from pyod.models.iforest import IForest
        clf = IForest()
    elif model == 'knn':
        from pyod.models.knn import KNN
        clf = KNN()
    elif model == 'lmdd':
        from pyod.models.lmdd import LMDD
        clf = LMDD()
    elif model == 'loci':
        from pyod.models.loci import LOCI
        clf = LOCI()
    elif model == 'loda':
        from pyod.models.loda import LODA
        clf = LODA()
    elif model == 'lof':
        from pyod.models.lof import LOF
        clf = LOF()
    elif model == 'mcd':
        from pyod.models.mcd import MCD
        clf = MCD()
    elif model == 'ocsvm':
        from pyod.models.ocsvm import OCSVM
        clf = OCSVM()
    elif model == 'pca':
        from pyod.models.pca import PCA
        clf = PCA()
    elif model == 'sod':
        from pyod.models.sod import SOD
        clf = SOD()
    elif model == 'vae':
        from pyod.models.vae import VAE
        clf = VAE()
    elif model == 'xgbod':
        from pyod.models.xgbod import XGBOD
        clf = XGBOD()
    else:
        raise ValueError(f"unknown model {model}")
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

