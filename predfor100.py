#predict next 7 days' hourly aggregated time series
#load model from S3
import boto3
import re
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Hide messy TensorFlow warnings
import tensorflow as tf
import numpy as np
from dep import metric_dict,reversed_metric_dict,reversed_dayofweek_dict,dayofweek_dict,df_acc_dict,acc_dict,acc_int_dict
import dep

s3 = boto3.resource('s3')
bucket=s3.Bucket('camp-neuralnet-model-prod')
prefix='prediction/top100-8metrics/'
client = boto3.client('s3')
paginator = client.get_paginator('list_objects')
results = paginator.paginate(Bucket='camp-neuralnet-model-prod', Delimiter='/',Prefix=prefix)
for result in results:
  for file in result.get('Contents'):
    print file.get('Key')
    key=file.get('Key')
    filename=re.sub('^'+prefix, '', key)
    bucket.download_file(file.get('Key'), './modelfile/'+filename)

csvdir_data='./csvdata/allacc8metrics_synth_valid.csv' 

df_input_raw = pd.read_csv(csvdir_data,parse_dates=True,index_col=['accountid_idx','metric_idx','hourstamp'],header=0, skipinitialspace=True,engine='c')
print 'load csv file finish'

idx=pd.IndexSlice


with tf.Session() as sess:
  save_path='./modelfile/0'
  graphfile_path=save_path+'.meta' 
  new_saver=tf.train.import_meta_graph(graphfile_path)
  new_saver.restore(sess,save_path)
  pred=tf.get_collection('pred')[0]
  sMAPError=tf.get_collection('sMAPError')[0]

  accountid_inputs=tf.get_collection('accountid_inputs')[0]
  dayofweek_inputs=tf.get_collection('dayofweek_inputs')[0]

  metric_inputs=tf.get_collection('metric_inputs')[0]
  ts_inputs=tf.get_collection('ts_inputs')[0]
  train_labels=tf.get_collection('train_labels')[0]

# valid 100 accounts in loop
  for accountid in acc_dict.keys():
    df_valid_1account=df_input_raw.loc[idx[acc_int_dict[accountid],:,:],:]
  
    batch_ts,batch_account_int,batch_metric,batch_dayofweek,batch_labels=dep.generate_batch(df_valid_1account,df_valid_1account.shape[0],data_index=0)
    #batch_accountid=batch_accountid.as_matrix()
    #batch_ts=batch_ts.as_matrix()
    #batch_dayofweek=batch_dayofweek.as_matrix()
    #batch_metric=batch_metric.as_matrix()
    #batch_labels=batch_labels.as_matrix()
    feed_input = {ts_inputs: batch_ts,accountid_inputs:batch_account_int,metric_inputs:  batch_metric,dayofweek_inputs:batch_dayofweek}
    predictions=pred.eval(feed_input)
    
    feed_valid=feed_input
    feed_valid[train_labels] = batch_labels
    predict_SMAPE=sess.run(sMAPError,feed_dict=feed_valid)
    accountname=acc_dict[accountid]
    print 'SMAPE:',predict_SMAPE,'  ',accountname,'  predictions shape:',predictions.shape
