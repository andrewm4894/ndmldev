#%%
import json

import pandas as pd

#%%

df = pd.DataFrame(
    [
        [1, 'x', 0.2346], [2, 'y', 1.2345],
    ],
    columns=['a', 'b', 'f']
)
print(df.round(2))

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