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
csvraw = pd.read_csv(csvdir, names=COLUMNS, dtype={'value': np.float64}, parse_dates=True,
  date_parser=dateparse_fn,index_col='fiveminstamp',header=0, skipinitialspace=True)

dta=csvraw["value"]

idx = pd.Index(pd.date_range('2016-01-01', '2017-05-02',freq='5min',closed='left'))
#del dta["YEAR"]

dta= dta.reindex(idx, fill_value=0)
#dta.to_csv("telit5minstamp_synth.csv",header = ["datausage"],index_label=["fiveminstamp"])


#diff=fitdata.diff()

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat
def smape(yhat,y):
  sMAPE=200*np.mean(np.abs(yhat-y)/(np.abs(yhat)+np.abs(y)))
  return sMAPE

fitdata=dta['2016-01-01':'2016-02-24']  #8 weeks data
#--------------------------seasonal arima------------------------
def custom_resampler(array_like):
  return array_like[0]
fitdata_raw=dta[:'2016-04-21']
fitdata2=fitdata_raw.apply(np.log) #16 weeks history data

observ = dta['2016-04-22']
mod_list=list()
pred_steps=288
predict=np.empty(pred_steps)
average_forcast=np.empty(pred_steps)
for i in range (pred_steps):
  fitdata_inloop=fitdata2[i:].resample('7D').apply(custom_resampler)
  fitdata_inloop2=fitdata_raw[i:].resample('7D').apply(custom_resampler)
  average_forcast[i]=fitdata_inloop2.mean()
  try:
    arima_mod = sm.tsa.ARIMA(fitdata_inloop,(3,0,0)).fit(disp=False)
    mod_list.append(arima_mod)
    nextpoint=arima_mod.forecast(steps=1)[0]
    predict[i]=nextpoint
    print('predicted in %sth data point: %.3f' % (i,nextpoint))
  except:
    predict[i]=average_forcast[i]
    print('No convergence on %sth data point, use average forcast instead: %.3f' % (i,predict[i]))
predict=np.exp(predict)
observ_slice=observ[:pred_steps]
sMAPE_seasonal=smape(observ_slice,predict)
sMAPE_average=smape(observ_slice,average_forcast)
print('Seasonal arima sMAPE: %.3f,    average forcast sMAPE: %.3f' % (sMAPE_seasonal,sMAPE_average))

#--------------------seasonal arima end line----------------------



#ar_coef, ma_coef = arma_mod2016.arparams, arma_mod2016.maparams
#resid = arma_mod2016.resid
#yhat = predict(ar_coef, fitdata) + predict(ma_coef, resid) # manual forecasting
#-----------non seasonal mult step forcasting -----------------------
arima_mod = sm.tsa.ARIMA(fitdata,(3,0,3)).fit(disp=False)
max_steps=288
smape_list=list()
for steps in range(max_steps):
  steps=steps+1  # start from 1
  predict=arima_mod.forecast(steps=steps)
  obs = dta['2016-02-25'][:steps]

  sMAPE=smape(obs,predict[0])
  smape_list.append(sMAPE)
  print('Test sMAPE: %.3f when forcasting %s steps' % (sMAPE,steps))
#-------------multistep end line---------------------------------
#for i in range(len(predict[0])):
#  print('>predicted=%.3f, expected=%.3f' % (predict[0][i], obs[i]))

fig = plt.figure(figsize=(24,8))
ax = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)
#ax = dta.ix['2016-01-28':'2016-01-29'].plot(ax=ax)
#ax2 = dta.ix['1700':].plot(ax=ax2)
fig = arima_mod.plot_predict('2016-01-29','2016-01-29', dynamic=True, ax=ax,plot_insample=False)
#fig = arma_mod30.plot_predict('1900', '2008', dynamic=True, ax=ax2,plot_insample=False)

plt.show()

