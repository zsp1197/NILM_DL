# Created by zhai at 2018/9/20
# Email: zsp1197@163.com
import pandas as pd
from datetime import datetime
import numpy as np
import Tools
store=pd.HDFStore('ampds.hdf5')
df=store['/submeters']
apps=list(df.columns)
print(apps)
# vid apps
for app in apps:
    pass