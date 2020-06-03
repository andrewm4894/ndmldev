#%%

from get_data import get_chart_df

#%%


#%%

df = get_chart_df('system.cpu', host='34.75.216.243:19999', after=1591184028, before=1591184889)

#%%

start = 1591184028
print(df.query(f'{start} <= time_idx <= 1591184040'))

#%%