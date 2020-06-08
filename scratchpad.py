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