from datetime import datetime
from elasticsearch import Elasticsearch

import certifi

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









































































