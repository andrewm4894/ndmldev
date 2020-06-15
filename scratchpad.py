#%%
import json
from collections import Counter, OrderedDict

import pandas as pd

#%%

df = pd.DataFrame(
    [
        [1, 0.2346], [1, 1.2345], [1.1, 1.2345]
    ],
    columns=['a', 'b']
)
print(df.std())
#%%

df.nunique() / len(df)

#%%

charts = ['system.cpu', 'system.load', 'a.b', 'foo.1', 'foo.2', 'foo.3']
charts

#%%

counts = OrderedDict(Counter([c.split('.')[0] for c in charts]).most_common())
counts = '|'.join([f"{c}:{counts[c]}" for c in counts])

#%%

df.loc[:, df.nunique() / len(df) > 0.05]

#%%

for i, row in df.iterrows():
    #print(f"{row.to_dict()[x] for x in row.to_dict()}")
    print()

#%%

import numpy as np

arr = np.array([
    [10, 200],
    [20, 300],
    [30, 400],
    [40, 500],
    [50, 600],
]
)

n_lags = 3
arr_orig = np.copy(arr)
for n_lag in range(1, n_lags+1):
    arr = np.concatenate((arr, np.roll(arr_orig, n_lag, axis=0)), axis=1)
arr = arr[n_lags:]

print(arr)

#%%

{
    "model": {
        "type": "hbos",
        "params": {
            "contamination": 0.1
        },
        "n_lags": 2
    },
    "return_type": "html"
}

#%%