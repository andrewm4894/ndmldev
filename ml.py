import numpy as np
from pyod.models.hbos import HBOS as PyODModel


def do_pyod(chart_cols, arr_baseline, arr_highlight):
    results = []
    for chart in chart_cols:
        model = PyODModel(contamination=0.01)
        model.fit(arr_baseline[:, chart_cols[chart]])
        anomaly_preds = model.predict(arr_highlight[:, chart_cols[chart]])
        anomaly_probs = model.predict_proba(arr_highlight[:, chart_cols[chart]])[:, 1]
        results.append([chart, np.mean(anomaly_probs), np.mean(anomaly_preds)])
    return results