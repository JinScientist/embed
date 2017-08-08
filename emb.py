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
df_train=df_train[df_train.index<'2017-06-04']

#df_train.loc['2017-05-02 23:00:00']
#df_train['2017-05']

#create dictionary for metric colomn:
metric_dict = dict()
for metric in df_train['metric'].unique():
  metric_dict[metric] = len(metric_dict)


df_train.insert(3,'metric_dict',df_train['metric'].map(metric_dict))
ds=df_train.set_index(['metric_dict',df_train.index]).loc[:,['value']]

iterables=[metric_dict.values(),pd.date_range('2017-02-01', '2017-06-04',freq='1h',closed='left')]
index= pd.MultiIndex.from_product(iterables,names=['metric_dict','hourstamp'])
df_sy= ds.reindex(index, fill_value=0)
df_sy['metric_dict']=df_sy.index.get_level_values(0)
df_sy['dayofweek']=df_sy.index.get_level_values(1).dayofweek

idx=pd.IndexSlice
df_sy.sort_index(inplace=True)

df_sy.loc[(0,'2017-03-20 04')]  # check missing value


df_empty=pd.DataFrame(data=np.zeros([df_sy.index.size,672+168]),index=df_sy.index)
ser=df_sy.loc[:,['metric_dict','value']]
#concate input data points
for i in range(672+168):
  s_inloop=ser.set_index(['metric_dict',ser.index.get_level_values(1)-pd.Timedelta(hours=i)])
  #if i < 671:
    #s_inloop.columns=[('h%s' % (i+1))]    
  #else: 
  #  s.columns=[('p%s' % (i-671))]  
  df_empty[i]=s_inloop
  print df_empty.loc[(0,'2017-02-01')]
df=pd.concat([df_sy,df_empty],axis=1)
print df.loc[(0,'2017-02-01')]
# slice to filter out NaN

df_sample=df.loc[idx[:,slice('2017-02-01 00','2017-04-30 00')],:]

# shows a gap in the raw data
df_sample2=df.loc[(0,slice('2017-01-30 00','2017-01-31 23')),:]

with graph.as_default():

  # Input data.
  categorical_inputs = tf.placeholder(tf.int32, shape=[batch_size,2])
  continious_inputs = tf.placeholder(tf.int32, shape=[batch_size,672])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 169])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    metric_embeddings = tf.Variable
        tf.random_uniform([len(metric_dict), embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, categorical_inputs[:,1])
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

