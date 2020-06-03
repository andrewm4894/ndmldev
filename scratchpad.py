#%%

import requests
import numpy as np

chart = 'system.cpu'
host='34.75.216.243:19999'
after=1591184028
before=1591184889

url = f"http://{host}/api/v1/data?chart={chart}&after={after}&before={before}&format={format}"
r = requests.get(url)
r_json = r.json()
print(r_json['data'])
data = np.array(r_json['data'], dtype=[(col, '<f4') for col in r_json['labels']])

#%%

[(col, '<f4') for col in r_json['labels']]

#%%

#%%