

def try_fit(clf, colname, data, default_model):
    try:
        clf.fit(data)
        result = 'success'
    except Exception as e:
        log.warning(e)
        log.info(f'... could not fit model for {colname}, trying default')
        clf = default_model()
        clf.fit(data)
        result = 'default'
    return clf, result


def init_counters(colnames):
    n_charts, n_dims = len(set([colname.split('|')[0] for colname in colnames])), len(colnames)
    n_bad_data, fit_success, fit_default, fit_fail = 0, 0, 0, 0
    return n_charts, n_dims, n_bad_data, fit_success, fit_default, fit_fail