import logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pyod.models.pca import PCA as DefaultPyODModel
import stumpy
from adtk.detector import InterQuartileRangeAD as ADTKDefault
from utils import summary_info, get_col_map, add_lags


log = logging.getLogger(__name__)

chart_level_models = pyod_models_supported + adtk_models_supported


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








