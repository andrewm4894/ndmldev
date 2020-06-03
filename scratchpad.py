#%%

from get_data import get_chart_df, get_chart_arr

#%%


#%%

#df = get_chart_df('system.cpu', host='34.75.216.243:19999', after=1591184028, before=1591184889)
names, arr = get_chart_arr('system.cpu', host='34.75.216.243:19999', after=1591184028, before=1591184889)

#%%

arr

#%%

names

#%%