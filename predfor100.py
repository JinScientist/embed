#predict next 7 days' hourly aggregated time series
#load model from S3
import boto3
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Hide messy TensorFlow warnings
import tensorflow as tf
import numpy as np


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

#batch_ts,batch_accountid,batch_metric,batch_dayofweek,batch_labels = generate_batch(
#        df_shuffle,batch_size,data_index)


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

  #print MODELNAME+str(subnum)+'  restored-->',' input dimention:',x.shape,', output dimention:',pred.shape
  feed_input = {ts_inputs: batch_ts, accountid_inputs: batch_accountid,metric_inputs: batch_metric, dayofweek_inputs:batch_dayofweek}
  feed_valid=feed_input.update({train_labels:batch_labels})
  predictions=pred.eval(feed_input)
  #print 'predictions shape:',predictions.shape
  
  predict_SMAPE=sess.run(sMAPError,feed_dict=feed_valid)
  tspredict=np.concatenate((tspredict,predictions),axis=1)

  SMAPEfinal=np.mean(predict_SMAPE)
  print "Final SMAPE:",SMAPEfinal


