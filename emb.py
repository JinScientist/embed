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
  
#The csv file was already synthesized by run 'synth9metrics.py'  
csvdir='./csvdata/telit9metrics_synth.csv' 

df_train = pd.read_csv(csvdir,parse_dates=True,index_col=['metric','hourstamp'],header=0, skipinitialspace=True)

#df_train=df_train.loc[8]


df_shuffle=df_train.sample(n=df_train.shape[0])

lag_columns=list(np.arange(672).astype(str))
future_columns=list((np.arange(168)+672).astype(str))



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

embedding_size = 16  # Dimension of the embedding vector
relu_size=8         # relu layer hidden node number connecting to embedding features
lag_size=672        # input window size of time series
sig_size=lag_size+relu_size
step_size=168
batch_size=1000
graph = tf.Graph()
with graph.as_default():

  # Input data.
  dayofweek_inputs=tf.placeholder(tf.int32, shape=[batch_size])
  metric_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  ts_inputs = tf.placeholder(tf.float32, shape=[None,lag_size])
  train_labels = tf.placeholder(tf.float32, shape=[None, step_size])
  metric_dataset = tf.constant(np.arange(9),dtype=tf.int32) # for valid the embeding learning

  # Ops and variables pinned to the CPU or GPU
  with tf.device('/cpu:0'):
    #logarithm normlization
    ts_inputs_log=tf.log(ts_inputs)   #avoid log on algorithm
    train_labels_log=tf.log(train_labels)

    # Look up embeddings for inputs.
    metric_embeddings = tf.Variable(tf.random_uniform([9, embedding_size], -1.0, 1.0,dtype=tf.float32))
    metric_embed = tf.nn.embedding_lookup(metric_embeddings, metric_inputs)

    dayofweek_embeddings = tf.Variable(tf.random_uniform([7, embedding_size], -1.0, 1.0,dtype=tf.float32))
    dayofweek_embed = tf.nn.embedding_lookup(dayofweek_embeddings, dayofweek_inputs)
    embed=tf.concat([metric_embed,dayofweek_embed],1)

  with tf.name_scope('relu1'):  
    relu1=perc(embed,embedding_size*2,relu_size,act_func='relu')
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
  norm = tf.sqrt(tf.reduce_sum(tf.square(metric_embeddings), 1, keep_dims=True))
  normalized_embeddings = metric_embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, metric_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

#function to generate training batch from pd frame: df_sample
data_index = 0
def generate_batch(batch_size):
  global data_index
  assert data_index < df_shuffle.shape[0],"data_index:%s is larger than sample data size" % data_index
  batch_all=df_shuffle[data_index:data_index+batch_size]
  batch_ts=batch_all.loc[:,lag_columns]  
  batch_metric=batch_all.loc[:,'metric_dict']
  batch_dayofweek=batch_all.loc[:,'dayofweek']
  batch_labels=batch_all.loc[:,future_columns]
  data_index=data_index+batch_size
  return batch_ts, batch_metric,batch_dayofweek,batch_labels

num_steps = 19

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  for step in xrange(num_steps):
    batch_ts, batch_metric,batch_dayofweek,batch_labels = generate_batch(
        batch_size)                  # generate different batch data for each training step? 
    feed_dict = {ts_inputs: batch_ts, metric_inputs: batch_metric, dayofweek_inputs:batch_dayofweek, train_labels: batch_labels}
    batch_smape=session.run(sMAPError,feed_dict=feed_dict)
    print 'sMAPE at batch number',step, '  is', batch_smape
    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    for sub_step in xrange(1000):
      _,loss_val = session.run([optimizer_ADAM,l2loss], feed_dict=feed_dict)
      if sub_step % 100 == 0:
        print 'loss at step ', step, '  sub-step', sub_step,': ', loss_val


    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    #if step % 10000 == 0:
    #  sim = similarity.eval()
    #  for i in xrange(valid_size):
    #    valid_word = reverse_dictionary[valid_examples[i]]
    #    top_k = 8  # number of nearest neighbors
    #    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    #    log_str = 'Nearest to %s:' % valid_word
    #    for k in xrange(top_k):
    #      close_word = reverse_dictionary[nearest[k]]
    #      log_str = '%s %s,' % (log_str, close_word)
    #    print(log_str)
  final_embeddings = normalized_embeddings.eval()

train_labels_log = session.run(train_labels_log, feed_dict=feed_dict)




#p1=p1.set_index(['metric_dict',p1.index-pd.Timedelta(hours=1)])


#for i in metric_dict.values():
#	aaa=df_train2.loc[i].loc[df_train2.loc[i].index+pd.Timedelta(hours=1)].iloc[:-1]
#	aaa=aaa.set_index(aaa.index-pd.Timedelta(hours=1))	
#	bbb=df_train2.loc[i].iloc[:-1]
#  
#
#
#df.loc[idx[:,'2017-2-28'],:]

#
#df_train['2017-02']
#
#df_train.loc[0, pd.to_datetime('2017-02-01 00:00:00',format='%Y-%m-%d %H')]
#
#df_train2.loc[:].loc[pd.to_datetime('2017-02-01 00:00:00',format='%Y-%m-%d %H'),'value']

