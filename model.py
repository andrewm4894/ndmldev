import logging

from model_adtk import do_adtk, adtk_models_supported
from model_ks import do_ks
from model_mp import do_mp, mp_models_supported
from model_pyod import do_pyod, pyod_models_supported

log = logging.getLogger(__name__)


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








