import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Hide messy TensorFlow warnings
import tensorflow as tf
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import pandas as pd
import datetime
  
#The csv file was already synthesized by SQL script  
csvdir='./csvdata/telit9metric.csv' 

COLUMNS = ["hourstamp", "week", "day_of_week", "metric","value"]
def dateparse_fn (timestampes):    
  return pd.to_datetime(timestampes,format='%Y-%m-%d %H')
df_train = pd.read_csv(csvdir, names=COLUMNS,parse_dates=True,
  date_parser=dateparse_fn,index_col='hourstamp',header=0, skipinitialspace=True)


# avoid zero value for logrithm normalization later
df_train['value']=df_train['value'].replace(0,np.finfo(np.float32).eps)

#create dictionary for metric colomn:
metric_dict = dict()
for metric in df_train['metric'].unique():
  metric_dict[metric] = len(metric_dict)


df_train.insert(3,'metric_dict',df_train['metric'].map(metric_dict))
ds=df_train.set_index(['metric_dict',df_train.index]).loc[:,['value']]

iterables=[metric_dict.values(),pd.date_range('2017-02-01', '2017-06-04',freq='1h',closed='left')]
index= pd.MultiIndex.from_product(iterables,names=['metric','hourstamp'])
df_sy= ds.reindex(index, fill_value=np.finfo(np.float32).eps)
df_sy['metric_dict']=df_sy.index.get_level_values(0)
df_sy['dayofweek']=df_sy.index.get_level_values(1).dayofweek

idx=pd.IndexSlice
df_sy.sort_index(inplace=True)

#df_sy.loc[(0,'2017-03-20 04')]  # check missing value


# transorm to Neural Nets features
df_empty=pd.DataFrame(data=np.zeros([df_sy.index.size,672+168]),index=df_sy.index)
ser=df_sy.loc[:,['value']]
#concate input data points
for i in range(672+168):
  s_inloop=ser.set_index([ser.index.get_level_values(0),ser.index.get_level_values(1)-pd.Timedelta(hours=i)])
  #if i < 671:
    #s_inloop.columns=[('h%s' % (i+1))]    
  #else: 
  #  s.columns=[('p%s' % (i-671))]  
  df_empty[i]=s_inloop
  print df_empty.loc[(0,'2017-02-01')]
df=pd.concat([df_sy,df_empty],axis=1)
print df.loc[(0,'2017-02-01')]
# slice to filter out NaN

#all 9 metric has full tracked ts data with this time window
df_sample=df.loc[idx[:,slice('2017-02-01 00','2017-04-30 00')],:]   

df_sample.to_csv("./csvdata/telit9metrics_synth.csv")