import numpy as np
import csv

csvdir_acc_dict='./csvdata/acc100_dict.csv' 

reader = csv.reader(open(csvdir_acc_dict))

acc_dict= dict()
for row in reader:
  key = row[0]
  acc_dict[key] = row[1]

acc_dict.pop('accountid')

#create dictionary for account_int colomn;
acc_int_dict = dict()
for accountid in acc_dict.keys():
  acc_int_dict[accountid] = len(acc_int_dict)
#print acc_int_dict

metric_dict ={'countUL': 0, 'imsiSum': 1, 'countCL': 2, 'deletePdpCountV1': 3, 'createPdpCountV1': 4,'mapsum': 5,'countSAI': 6, 'countUGL': 7}

reversed_metric_dict = dict(zip(metric_dict.values(), metric_dict.keys()))

dayofweek_dict={'Monday': 0, 'Tuesday': 1, 'Wednsday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
reversed_dayofweek_dict = dict(zip(dayofweek_dict.values(), dayofweek_dict.keys()))
lag_columns=list(np.arange(672).astype(str))
future_columns=list((np.arange(168)+672).astype(str))
def generate_batch(data,batch_size,data_index):
  assert data_index < data.shape[0],"data_index:%s is larger than sample data size" % data_index
  batch_all=data[data_index:data_index+batch_size]
  batch_ts=batch_all.loc[:,lag_columns]
  batch_accountid=batch_all.loc[:,'account_int']
  batch_metric=batch_all.loc[:,'metric']
  batch_dayofweek=batch_all.loc[:,'dayofweek']
  batch_labels=batch_all.loc[:,future_columns]
  return batch_ts,batch_accountid,batch_metric,batch_dayofweek,batch_labels
