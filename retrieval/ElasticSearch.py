from elasticsearch import Elasticsearch

def setting():
    es = Elasticsearch('localhost:9200')
