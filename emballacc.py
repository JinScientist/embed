#train with accountid, metric and dayofweek as embedding features.
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
from dep import metric_dict,reversed_metric_dict,reversed_dayofweek_dict,dayofweek_dict,df_acc_dict,acc_dict
  
#The csv file was already synthesized by run 'synth9metrics.py'  
csvdir_data_train='./csvdata/allacc8metrics_synth_train.csv' 
csvdir_data_valid='./csvdata/allacc8metrics_synth_valid.csv' 

print df_acc_dict[df_acc_dict['accountid']==135462906]['accountname']  # use this instead as dict() has encoder problems

df_data_train = pd.read_csv(csvdir_data_train,parse_dates=True,index_col=['accountid_idx','metric_idx','hourstamp'],header=0, skipinitialspace=True,engine='c')
df_data_valid = pd.read_csv(csvdir_data_valid,parse_dates=True,index_col=['accountid_idx','metric_idx','hourstamp'],header=0, skipinitialspace=True,engine='c')

#df_train=df_train.loc[8]
idx=pd.IndexSlice
#df_train=df_data_train.loc[idx[:,:,slice('2017-05-01 00','2017-05-07 23')],:]

df_valid=df_data_valid.loc[idx[109351305,:,:],:] # Telit
#df_valid=df_data.loc[idx[118035406,:,slice('2017-07-01 00','2017-07-07 23')],:] # Latvi Energo
#df_valid=df_data.loc[idx[:,:,slice('2017-07-01 00','2017-07-07 23')],:]

df_shuffle=df_data_train.sample(n=df_data_train.shape[0])

lag_columns=list(np.arange(672).astype(str))
future_columns=list((np.arange(168)+672).astype(str))


emsize_accountid =7  # Dimension of the embedding vector for accountid
emsize_metric = 4  # Dimension of the embedding vector for metric
emsize_dayofweek =4  # Dimension of the embedding vector for 
embeddings_size=emsize_accountid+emsize_metric+emsize_dayofweek
relu_size=12         # relu layer hidden node number connecting to embedding features
lag_size=672        # input window size of time series
sig_size=lag_size+relu_size
step_size=168
batch_size=26880  # total sample size 134400 per week
num_steps = 5
epoch=2000

def perc (input,size_in,size_out,act_func,name="perc"):
    with tf.name_scope('weights'):
      w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
      #variable_summaries(w)   
    with tf.name_scope('biases'):
      b = tf.Variable(tf.constant(0.1, shape=[size_out]),name="B")
      #variable_summaries(b)
    preactivate=tf.matmul(input,w)+b
    if act_func=='sigmoid':
      act =tf.nn.sigmoid(preactivate)
    elif act_func=='relu':
      act =tf.nn.relu(preactivate)
    elif act_func=='tanh':
      act =tf.nn.tanh(preactivate)
    else: raise ValueError('activation function not recogonized')
    tf.summary.histogram("preactivate",preactivate)
    tf.summary.histogram("activations",act)
    return act
def regr_out (input,size_in,size_out,name="regr_output"):
    with tf.name_scope('weights'):
      w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='WOUT')
      #variable_summaries(w)   
    with tf.name_scope('biases'):
      b = tf.Variable(tf.constant(0.1, shape=[size_out]),name="BOUT")
      #variable_summaries(b)
    output=tf.matmul(input,w)+b
    return output


graph = tf.Graph()
with graph.as_default():

  # Input data.
  accountid_inputs=tf.placeholder(tf.int32, shape=[None])
  dayofweek_inputs=tf.placeholder(tf.int32, shape=[None])
  metric_inputs = tf.placeholder(tf.int32, shape=[None])
  ts_inputs = tf.placeholder(tf.float32, shape=[None,lag_size])
  train_labels = tf.placeholder(tf.float32, shape=[None, step_size])

  accountid_dataset = tf.constant(df_acc_dict['accountid'].values,dtype=tf.int32) # for valid the embeding learning
  metric_dataset = tf.constant(np.arange(8),dtype=tf.int32)
  dayofweek_dataset=tf.constant(np.arange(7),dtype=tf.int32)


  #logarithm normlization
  ts_inputs_log=tf.log(ts_inputs)   #avoid log on algorithm
  train_labels_log=tf.log(train_labels)
  # Look up embeddings for inputs.
  accountid_embeddings = tf.Variable(tf.random_uniform([100, emsize_accountid], -1.0, 1.0,dtype=tf.float32))
  accountid_embed = tf.nn.embedding_lookup(accountid_embeddings, accountid_inputs)
  metric_embeddings = tf.Variable(tf.random_uniform([8, emsize_metric], -1.0, 1.0,dtype=tf.float32))
  metric_embed = tf.nn.embedding_lookup(metric_embeddings, metric_inputs)
  dayofweek_embeddings = tf.Variable(tf.random_uniform([7, emsize_dayofweek], -1.0, 1.0,dtype=tf.float32))
  dayofweek_embed = tf.nn.embedding_lookup(dayofweek_embeddings, dayofweek_inputs)
  embed=tf.concat([accountid_embed,metric_embed,dayofweek_embed],1)

  with tf.name_scope('relu1'):  
    relu1=perc(embed,embeddings_size,relu_size,act_func='relu')
  with tf.name_scope('relu2'):  
    relu2=perc(relu1,relu_size,relu_size,act_func='relu')
  ts_with_embed=tf.concat([relu2,ts_inputs_log],1)

  with tf.name_scope('sig_layer'):
    sig=perc(ts_with_embed,sig_size,sig_size,act_func='sigmoid')
  with tf.name_scope('output_layer'):  
    target=regr_out(sig,sig_size,step_size)
  with tf.name_scope("loss"):
    l2loss=tf.nn.l2_loss(train_labels_log-target)
    tf.summary.scalar("l2loss",l2loss)
  with tf.name_scope("target_unnorm"):
    target_unnorm=tf.exp(target)
  with tf.name_scope("SMAPE"):
    diff = tf.abs((train_labels- target_unnorm) / tf.maximum((tf.abs(train_labels)+tf.abs(target_unnorm)), 1e-08))
    sMAPError=200*tf.reduce_mean(diff)

  global_step = tf.Variable(0, trainable=False)
  with tf.name_scope("train"):
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer_SGD = tf.train.GradientDescentOptimizer(1.0).minimize(l2loss)
    optimizer_ADAM=tf.train.AdamOptimizer(1E-4).minimize(l2loss,global_step=global_step)
  # Compute the cosine similarity between valid dataset and all embeddings.


  # metric embeddings
  norm_metric = tf.sqrt(tf.reduce_sum(tf.square(metric_embeddings), 1, keep_dims=True))
  normalized_embeddings_metric = metric_embeddings / norm_metric
  valid_embeddings_metric = tf.nn.embedding_lookup(
      normalized_embeddings_metric, metric_dataset)
  similarity_metric = tf.matmul(
      valid_embeddings_metric, normalized_embeddings_metric, transpose_b=True)

  # dayofweek embeddings
  norm_dayofweek = tf.sqrt(tf.reduce_sum(tf.square(dayofweek_embeddings), 1, keep_dims=True))
  normalized_embeddings_dayofweek = dayofweek_embeddings / norm_dayofweek

  #valid_embeddings_dayofweek is the same data rows as normalized_embeddings_dayofweek
  # Just in different row order, valid_embeddings_dayofweek is in the order of metric from 0 to 7
  valid_embeddings_dayofweek = tf.nn.embedding_lookup(
      normalized_embeddings_dayofweek, dayofweek_dataset)
  similarity_dayofweek = tf.matmul(
      valid_embeddings_dayofweek, normalized_embeddings_dayofweek, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
  tf.add_to_collection('pred',target_unnorm)  
  tf.add_to_collection('sMAPError',sMAPError)

  tf.add_to_collection('accountid_inputs',accountid_inputs)
  tf.add_to_collection('dayofweek_inputs',dayofweek_inputs)
  tf.add_to_collection('metric_inputs',metric_inputs)

  tf.add_to_collection('ts_inputs',ts_inputs)
  tf.add_to_collection('train_labels',train_labels)

def generate_batch(data,batch_size,data_index):
  assert data_index < data.shape[0],"data_index:%s is larger than sample data size" % data_index
  batch_all=data[data_index:data_index+batch_size]
  batch_ts=batch_all.loc[:,lag_columns]
  batch_accountid=batch_all.loc[:,'accountid']
  batch_metric=batch_all.loc[:,'metric']
  batch_dayofweek=batch_all.loc[:,'dayofweek']
  batch_labels=batch_all.loc[:,future_columns]
  return batch_ts,batch_accountid,batch_metric,batch_dayofweek,batch_labels


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')
  data_index = 0
  #pre generate the validation batch
  valid_ts,valid_accountid, valid_metric,valid_dayofweek,valid_labels = generate_batch(
    df_valid,batch_size=df_valid.shape[0],data_index=0)
  feed_valid = {ts_inputs: valid_ts, accountid_inputs: valid_accountid, metric_inputs: valid_metric, dayofweek_inputs:valid_dayofweek, train_labels: valid_labels}
  #Train start
  for step in xrange(num_steps):
    batch_ts,batch_accountid,batch_metric,batch_dayofweek,batch_labels = generate_batch(
        df_shuffle,batch_size,data_index)
    print 'batch number %s is generated' % step
    data_index=data_index+batch_size                  # generate different batch data for each training step? 
    feed_train = {ts_inputs: batch_ts, accountid_inputs: batch_accountid,metric_inputs: batch_metric, dayofweek_inputs:batch_dayofweek, train_labels: batch_labels}
    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    for sub_step in xrange(epoch):
      _,loss_val = session.run([optimizer_ADAM,l2loss], feed_dict=feed_train)
      if sub_step % 100 == 0:
        batch_smape=session.run(sMAPError,feed_dict=feed_valid)
        print 'loss at step ', step, '  sub-step', sub_step,': ', loss_val,'  sMAPE is ', batch_smape
        if batch_smape<10.5:break #break for each batch,make sure all batch is learnt


  #save the model
  MODELNAME='a100-8metrics'
  RUN=0
  checkpoint_path_trained='./models/'+MODELNAME+'/'
  if not os.path.exists(checkpoint_path_trained):
    os.makedirs(checkpoint_path_trained)
  saver=tf.train.Saver()
  save_path=saver.save(session,checkpoint_path_trained+str(RUN))
  print("Tained model %s saved in file:%s" % (MODELNAME,save_path))

  sim_metric = similarity_metric.eval()
  for i in xrange(8):
    valid_metric = reversed_metric_dict[i]
    top_k = 3  # number of nearest neighbors
    nearest = (-sim_metric[i, :]).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_metric
    for k in xrange(top_k):
      close_word = reversed_metric_dict[nearest[k]]
      log_str = '%s %s,' % (log_str, close_word)
    print(log_str)

  sim_dayofweek = similarity_dayofweek.eval()
  for i in xrange(7):
    valid_dayofweek = reversed_dayofweek_dict[i]
    top_k = 3  # number of nearest neighbors
    nearest = (-sim_dayofweek[i, :]).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_dayofweek
    for k in xrange(top_k):
      close_word = reversed_dayofweek_dict[nearest[k]]
      log_str = '%s %s,' % (log_str, close_word)
    print(log_str)
  #final_embeddings = normalized_embeddings.eval()







