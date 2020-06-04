#%%

import pandas as pd
df = pd.DataFrame(
    [
        ['2012', 'A', 3, 4], ['2012', 'B', 8, 5], ['2011', 'A', 20, 7], ['2011', 'B', 30, 3], ['2011', 'C', 40, 5],
    ],
    columns=['chart', 'dim', 'ks', 'p'])
#df['chart_rank'] = df.groupby(['chart'])[['ks']].agg('mean').rank()
print(df)

#%%

df_results_chart = df.groupby(['chart'])[['ks', 'p']].agg(['mean', 'min', 'max'])
df_results_chart.columns = ['_'.join(col) for col in df_results_chart.columns]
df_results_chart = df_results_chart.reset_index()
print(df_results_chart)

#%%

['_'.join(col) for col in df_results_chart.columns]

#%%