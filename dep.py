import pandas as pd
import numpy as np

csvdir_acc_dict='./csvdata/acc100_dict.csv' 

df_acc_dict = pd.read_csv(csvdir_acc_dict,header=0, skipinitialspace=True,
  dtype={"accountid": np.int32,'accountname':object},engine='c')

acc_dict= dict()
for i in range(100):
  acc_dict[df_acc_dict.loc[i]['accountid']] = df_acc_dict.loc[i]['accountname']

metric_dict ={'countUL': 4, 'imsiSum': 6, 'countCL': 1, 'createPdpCountV2': 0, 'createPdpCountV1': 5, 
'mapsum': 7,'countSAI': 2, 'countUGL': 3}

reversed_metric_dict = dict(zip(metric_dict.values(), metric_dict.keys()))

dayofweek_dict={'Monday': 0, 'Tuesday': 1, 'Wednsday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
reversed_dayofweek_dict = dict(zip(dayofweek_dict.values(), dayofweek_dict.keys()))
lag_columns=list(np.arange(672).astype(str))
future_columns=list((np.arange(168)+672).astype(str))
def generate_batch(data,batch_size,data_index):
  assert data_index < data.shape[0],"data_index:%s is larger than sample data size" % data_index
  batch_all=data[data_index:data_index+batch_size]
  batch_ts=batch_all.loc[:,lag_columns]
  batch_accountid=batch_all.loc[:,'accountid']
  batch_metric=batch_all.loc[:,'metric']
  batch_dayofweek=batch_all.loc[:,'dayofweek']
  batch_labels=batch_all.loc[:,future_columns]
  return batch_ts,batch_accountid,batch_metric,batch_dayofweek,batch_labels