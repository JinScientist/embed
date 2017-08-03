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
#print df_train

#df_train.loc['2017-05-02 23:00:00']
#df_train['2017-05']

#create dictionary for metric colomn:
metric_dict = dict()
for metric in df_train['metric'].unique():
  metric_dict[metric] = len(metric_dict)


df_train.insert(3,'metric_dict',df_train['metric'].map(metric_dict))

df_index=df_train.index+pd.Timedelta(hours=1)
p1=df_train.loc[df_index.get_values()]

df_train2=df_train.set_index(['metric_dict',df_train.index])
df=df_train2.copy()

#concate input data points
for i in range(672-1+168):
  s=df_train.set_index(['metric_dict',df_train.index-pd.Timedelta(hours=1+i)])['value']
  if i < 671:
    s=s.rename('h%s' % (i+1))     # input part 
  else: s=s.rename('p%s' % (i-671))   # target part
  df=pd.concat([df,s],axis=1)
  print df.loc[(0,'2017-02-01')]

# shows a gap in the raw data
df.loc[(1,slice('2017-01-30 00','2017-01-31 23')),:]

# slice to filter out NaN
df.loc[(slice(None),slice('2017-02-01 00','2017-04-30 00')),:]

#concatenate target data points
for i in range(168):
  p=df_train.set_index(['metric_dict',df_train.index-pd.Timedelta(hours=1+i)])['value']
  p=p.rename('p%s' % (i))
  df=pd.concat([df,p],axis=1)
  print df[i+672].head()


#p1=p1.set_index(['metric_dict',p1.index-pd.Timedelta(hours=1)])


#for i in metric_dict.values():
#	aaa=df_train2.loc[i].loc[df_train2.loc[i].index+pd.Timedelta(hours=1)].iloc[:-1]
#	aaa=aaa.set_index(aaa.index-pd.Timedelta(hours=1))	
#	bbb=df_train2.loc[i].iloc[:-1]
#  
#
#
df.loc[idx[:,'2017-2-28'],:]
idx = pd.IndexSlice
print df.columns
#
#df_train['2017-02']
#
#df_train.loc[0, pd.to_datetime('2017-02-01 00:00:00',format='%Y-%m-%d %H')]
#
#df_train2.loc[:].loc[pd.to_datetime('2017-02-01 00:00:00',format='%Y-%m-%d %H'),'value']

