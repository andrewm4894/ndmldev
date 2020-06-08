#%%

import pandas as pd

#%%

df = pd.DataFrame(
    [
        [1, 'x'], [2, 'y'],
    ],
    columns=['a', 'b']
)
print(df)

#%%

for i, row in df.iterrows():
    print(str(row.to_dict()))

#%%