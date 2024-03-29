import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Hide messy TensorFlow warnings
import tensorflow as tf
import sys
import shutil
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import pandas as pd
import datetime
import time
import boto3
import s3fs
from dep import acc_dict,metric_dict,acc_int_dict

RUN_SQL_FLAG= True
s3 = boto3.resource('s3')
bucketname='camp-neuralnet-model-prod'
bucket=s3.Bucket(bucketname)
clientAthena = boto3.client('athena','eu-west-1')
def query():
	bucket.objects.filter(Prefix="prediction/training-data").delete()

	with open('query_for_training.sql', 'r') as sqlfile:
		sql=sqlfile.read().replace('\n', ' ')
	print sql
	response = clientAthena.start_query_execution(
	    QueryString=sql,
	    ResultConfiguration={
	        'OutputLocation': 's3://camp-neuralnet-model-prod/prediction/training-data'
	    }
	)
	return response['QueryExecutionId']
def run_sql():
  if RUN_SQL_FLAG==True:
    query_exc_id=query()
    
    #wait until athena finish
    while True:
    	time.sleep(5)
    	queryState = clientAthena.get_query_execution(
       	QueryExecutionId=query_exc_id)['QueryExecution']['Status']['State']
    	if queryState== 'SUCCEEDED':
    		print 'query state:',queryState
    		break
    	print 'query state:',queryState
  
  for obj in bucket.objects.filter(Prefix='prediction/training-data'):
  	key=obj.key
  	break
  print key
  return key

key=run_sql()

#The csv file was already synthesized by SQL script  
csvdir='s3://'+bucketname+'/'+key 
COLUMNS = ["accountid","hourstamp", "day_of_week", "metric","value"]
def dateparse_fn (timestampes):    
  return pd.to_datetime(timestampes,format='%Y-%m-%d %H')
df_train = pd.read_csv(csvdir, names=COLUMNS,parse_dates=True,
  date_parser=dateparse_fn,index_col='hourstamp',header=0, skipinitialspace=True,dtype={"accountid": str})


# avoid zero value for logrithm normalization later
df_train['value']=df_train['value'].replace(0,np.finfo(np.float32).eps)

#metric_dict ={'countUL': 4, 'imsiSum': 6, 'countCL': 1, 'createPdpCountV2': 0, 'createPdpCountV1': 5, 'mapsum': 7,'countSAI': 2, 'countUGL': 3}
accountid_array=acc_dict.keys()

#make multi index with 3 layers
df_train.insert(3,'metric_dict',df_train['metric'].map(metric_dict))
df_train.insert(3,'account_int',df_train['accountid'].map(acc_int_dict))
ds=df_train.set_index(['account_int','metric_dict',df_train.index]).loc[:,['value']]

#fill up empty rows in the time series
iterables=[acc_int_dict.values(),metric_dict.values(),pd.date_range('2017-07-01', '2017-10-31',freq='1h',closed='left')]
index= pd.MultiIndex.from_product(iterables,names=['account_int_idx','metric_idx','hourstamp'])
df_sy= ds.reindex(index, fill_value=np.finfo(np.float32).eps)
df_sy['account_int']=df_sy.index.get_level_values(0)
df_sy['metric']=df_sy.index.get_level_values(1)
df_sy['dayofweek']=df_sy.index.get_level_values(2).dayofweek

idx=pd.IndexSlice
df_sy.sort_index(inplace=True)

#df_sy=df.loc[idx[:,:,slice('2017-05-01 00','2017-05-25 00')],:]
#df_sy.loc[(0,'2017-03-20 04')]  # check missing value


# transorm to Neural Nets features
df_empty=pd.DataFrame(data=np.zeros([df_sy.index.size,672+168]),index=df_sy.index)
ser=df_sy.loc[:,['value']]
#concate input data points
for i in range(672+168):
  s_inloop=ser.set_index([ser.index.get_level_values(0),ser.index.get_level_values(1),ser.index.get_level_values(2)-pd.Timedelta(hours=i)])
  #if i < 671:
    #s_inloop.columns=[('h%s' % (i+1))]    
  #else: 
  #  s.columns=[('p%s' % (i-671))]
  df_empty[i]=s_inloop
  if i%5==0:print 'reformat progress: ', "{0:.2f}%".format(i/840.0 * 100)
  #print df_empty.loc[(109351305,1,'2017-05-01')] # telit id
df=pd.concat([df_sy,df_empty],axis=1)
# slice to filter out NaN

#all 9 metric has full tracked ts data with this time window
df=df.drop('value',axis=1)
df_sample=df.loc[idx[:,:,slice('2017-07-01 00','2017-07-07 23')],:]   
df_valid=df.loc[idx[:,:,slice('2017-09-01 00','2017-09-07 23')],:]  

df_sample.to_csv("./csvdata/allacc8metrics_synth_train.csv")
df_valid.to_csv("./csvdata/allacc8metrics_synth_valid.csv")
