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
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]]
)

#%%



#%%

n_lags = 1

for n_lag in range(n_lags):
    arr = np.concatenate((arr, np.roll(arr, n_lag, axis=0)), axis=1)

#%%