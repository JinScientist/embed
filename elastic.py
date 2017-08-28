#pandas to elastic
from datetime import datetime
from elasticsearch import Elasticsearch
import certifi
import json
import requests
from synth9metrics import df_sy
import pandas as pd

idx=pd.IndexSlice
df=df_sy.loc[idx[:,slice('2017-02-01 00','2017-04-30 00')],:]
df.reset_index(inplace=True)
df_as_json = df.to_json(orient='records', lines=True,date_format='iso')

final_json_string = ''
for json_document in df_as_json.split('\n'):
    jdict = json.loads(json_document)
    metadata = json.dumps({'index': 
      {'_id': str(jdict['metric'])+'_'+str(jdict['hourstamp'])}
      })

    jdict.pop('metric')
    final_json_string += metadata + '\n' + json.dumps(jdict) + '\n'


headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

r = requests.post('https://c34d6f2275e9c550006404f8188c2e61.eu-west-1.aws.found.io:9243/index28/first-type/_bulk', auth=('elastic', 'Y7j8qXoqXP5FUuXzwCCckD39'),data=final_json_string, headers=headers, timeout=60)


print df_json

es = Elasticsearch(
    ['https://c34d6f2275e9c550006404f8188c2e61.eu-west-1.aws.found.io:9243'],
    http_auth=('elastic', 'Y7j8qXoqXP5FUuXzwCCckD39'),
    port=443,
    use_ssl=True
)


res = es.get(index="iot_index", doc_type='new_type', id=1)
print(res['_source'])

doc = {
    'author': 'kimchy',
    'text': 'Elasticsearch: cool. bonsai cool.',
    'timestamp': datetime.now(),
}
res = es.index(index="iot_index", doc_type='tweet', id=1, body=doc)
print(res['created'])

res = es.get(index="test-index", doc_type='tweet', id=1)
print(res['_source'])

es.indices.refresh(index="test-index")

res = es.search(index="test-index", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total'])
for hit in res['hits']['hits']:
    print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])






































































