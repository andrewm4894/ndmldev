#%%

from datetime import datetime

after = 1591100938000
now_ts = int(datetime.now().timestamp())

after_secs = now_ts - (after / 1000)
print(after_secs)

#%%
