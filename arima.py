from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(pd.date_range('1700', '2009',freq='A'))
del dta["YEAR"]
arma_mod20 = sm.tsa.ARMA(dta, (3,3)).fit(disp=False)
arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit(disp=False)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax = dta.ix['1950':].plot(ax=ax)
ax2 = dta.ix['1950':].plot(ax=ax2)
fig = arma_mod20.plot_predict('1990', '2012', dynamic=True, ax=ax,plot_insample=False)
fig = arma_mod30.plot_predict('1990', '2012', dynamic=True, ax=ax2,plot_insample=False)
plt.show()