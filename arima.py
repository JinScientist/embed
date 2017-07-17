from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

#dta = sm.datasets.sunspots.load_pandas().data
csvdir='./csvdata/telit5minstamp.csv'
COLUMNS = ["fiveminstamp","value"]
def dateparse_fn (timestampes):    
  return pd.to_datetime(timestampes,format='%Y-%m-%d %H:%M')
csvraw = pd.read_csv(csvdir, names=COLUMNS,parse_dates=True,
  date_parser=dateparse_fn,index_col='fiveminstamp',header=0, skipinitialspace=True)

dta=csvraw["value"]

idx = pd.Index(pd.date_range('2016-01-01', '2017-05-02',freq='5min',closed='left'))
#del dta["YEAR"]

dta= dta.reindex(idx, fill_value=0)
dta.to_csv("telit5minstamp_synth.csv",header = ["datausage"],index_label=["fiveminstamp"])
fitdata=dta[0:8*2016]
arma_mod2016 = sm.tsa.ARMA(fitdata, (2016,2016)).fit(disp=False)
#arma_mod30 = sm.tsa.ARMA(dta, (3,3)).fit(disp=False)
fig = plt.figure(figsize=(24,8))
ax = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)
ax = dta.ix['1700':].plot(ax=ax)
#ax2 = dta.ix['1700':].plot(ax=ax2)
fig = arma_mod20.plot_predict(8*2016,8*2016+288, dynamic=True, ax=ax,plot_insample=False)
#fig = arma_mod30.plot_predict('1900', '2008', dynamic=True, ax=ax2,plot_insample=False)
plt.show()
