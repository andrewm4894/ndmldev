import logging

from model_adtk import do_adtk
from model_ks import do_ks
from model_mp import do_mp, mp_models_supported
from model_pyod import pyod_models_supported, do_pyod

log = logging.getLogger(__name__)

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
adtk_meta_models = ['linear', 'rf', 'huber', 'knnad', 'kernridge']


def run_model(model, colnames, arr_baseline, arr_highlight):
    """Function to take in data and some config and decide what model to run.
    """
    if model['type'] in pyod_models_supported:
        results = do_pyod(model, colnames, arr_baseline, arr_highlight)
    elif model['type'] in mp_models_supported:
        results = do_mp(model['type'], colnames, arr_baseline, arr_highlight)
    elif model['type'] in adtk_models_supported:
        results = do_adtk(model, colnames, arr_baseline, arr_highlight)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)
    return results








