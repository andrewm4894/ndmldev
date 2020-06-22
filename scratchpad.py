#%%

n_lags = 2
cols = ['cola', 'colb']


[f"{col}_lag{n_lag}" for n_lag in range(n_lags) for col in cols]

#%%

# get max and min scores
results_list = []
for chart in results:
    for dimension in results[chart]:
        for k in dimension:
            results_list.append([chart, k, dimension[k]['score']])

print(results_list)

#%%
import pandas as pd

data = [['chart.a', 'dim.a', 0.167], ['chart.a', 'sent', 0.15], ['chart.a', 'delivered', 0.1678], ['chart.b', 'dim.a', 0.14394685039], ['chart.b', 'dim.b', 0.22]]
df = pd.DataFrame(data, columns=['chart', 'dimension', 'score'])
df['chart_rank'] = df['chart'].map(df.groupby('chart')[['score']].mean().rank()['score'].to_dict())
print(df)

#%%




#%%

#%%