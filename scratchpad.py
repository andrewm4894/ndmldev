#%%
from urllib.parse import urlparse

url_parse = urlparse("http://35.246.123.204:19999/host/london.my-netdata.io/#menu_system;after=1591368740000;before=1591369159000;highlight_after=1591368948297;highlight_before=1591369120515;theme=slate;help=true")

print(url_parse.path)

#%%

df_results_chart = df.groupby(['chart'])[['ks', 'p']].agg(['mean', 'min', 'max'])
df_results_chart.columns = ['_'.join(col) for col in df_results_chart.columns]
df_results_chart = df_results_chart.reset_index()
print(df_results_chart)

#%%

charts = ['system.cpu', 'system.load']
colnames = ['system.cpu__user', 'system.cpu__guest', 'system.load__load5', 'system.load__load15']

chart_cols = []
for chart in charts:
    chart_cols.append([colnames.index(col) for col in colnames if col.startswith(chart)])

print(chart_cols)

#%%