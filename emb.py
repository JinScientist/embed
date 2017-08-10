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

#all 9 metric has full tracked ts data with this time window
df_sample=df.loc[idx[:,slice('2017-02-01 00','2017-04-30 00')],:]   
df_shuffle=df_sample.sample(n=df_sample.shape[0])
lag_columns=list(np.arange(672))
future_columns=list(np.arange(168)+672)
df_shuffle[0:batch_size].loc[:,lag_columns]      
df_shuffle[0:batch_size].loc[:,future_columns]

def perc (input,size_in,size_out,act_func,name="perc"):
    with tf.name_scope('weights'):
      w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
      variable_summaries(w)   
    with tf.name_scope('biases'):
      b = tf.Variable(tf.constant(0.1, shape=[size_out]),name="B")
      variable_summaries(b)
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
      variable_summaries(w)   
    with tf.name_scope('biases'):
      b = tf.Variable(tf.constant(0.1, shape=[size_out]),name="BOUT")
      variable_summaries(b)
    output=tf.matmul(input,w)+b
    return output

embedding_size = 4  # Dimension of the embedding vector
relu_size=3         # relu layer hidden node number connecting to embedding features
lag_size=672        # input window size of time series
step_size=168
with graph.as_default():

  # Input data.
  dayofweek_inputs=f.placeholder(tf.int32, shape=[batch_size])
  metric_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  ts_inputs = tf.placeholder(tf.float32, shape=[batch_size,lag_size])
  train_labels = tf.placeholder(tf.float32, shape=[batch_size, step_size])
  metric_dataset = tf.constant(np.arange(len(metric_dict)),dtype=tf.int32) # for valid the embeding learning

  # Ops and variables pinned to the CPU or GPU
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    metric_embeddings = tf.Variable(tf.random_uniform([len(metric_dict), embedding_size], -1.0, 1.0))
    metric_embed = tf.nn.embedding_lookup(metric_embeddings, metric_inputs)

    dayofweek_embeddings = tf.Variable(tf.random_uniform([7, embedding_size], -1.0, 1.0))
    dayofweek_embed = tf.nn.embedding_lookup(embeddings, dayofweek_inputs)
    embed=tf.concat([metric_embed,dayofweek_embed],1)

  with tf.name_scope('relu1'):  
    relu1=perc(embed,embed.shape[1],relu_size,act_func='relu')
  with tf.name_scope('relu2'):  
    relu2=perc(relu1,relu_size,relu_size,act_func='relu')
  ts_with_embed=tf.concat([relu2,ts_inputs],1)

  with tf.name_scope('sig_layer'):
    sig=perc(ts_with_embed,relu_size+lag_size,relu_size+lag_size,act_func='sigmoid')
  with tf.name_scope('output_layer'):  
    target=regr_out(sig,relu_size+lag_size,step_size)
  with tf.name_scope("loss"):
    l2loss=tf.nn.l2_loss(train_labels-target)
    tf.summary.scalar("l2loss",l2loss)

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between valid dataset and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(metric_embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, metric_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

#function to generate training batch from pd frame: df_sample
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
num_steps = 100001







with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_ts, batch_metric,batch_dayofweek = generate_batch(
        batch_size)                  # generate different batch data for each training step? 
    feed_dict = {ts_inputs: batch_ts, metric_inputs: batch_metric, dayofweek_inputs:batch_dayofweek, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()






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

