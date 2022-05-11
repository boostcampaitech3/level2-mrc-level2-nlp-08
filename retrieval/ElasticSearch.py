from elasticsearch import Elasticsearch, helpers
from arguments import return_arg4ES
from contextlib import contextmanager
from datasets import Dataset, concatenate_datasets, load_from_disk
import time
import json
import os
from typing import List, NoReturn, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class ElasticSearch:
    def __init__(
        self,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        self.args = return_arg4ES()
        ELASTIC_PASSWORD = ""

        CLOUD_ID = "elastic:"

        # Create the client instance
        self.es = Elasticsearch(
                            cloud_id=CLOUD_ID,
                            basic_auth=("elastic", ELASTIC_PASSWORD),
                           timeout=self.args.es_timeout,
                           max_retries=self.args.es_max_retries,
                           retry_on_timeout=self.args.es_retry_on_timeout
                                )
        assert self.es.ping(), "Fail ping."
        self.index_name = self.args.index_name
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")

    def insert_data(self):
        """insert data to ElasticSearch"""
        index_name = self.index_name
        if not self.es.indices.exists(index=index_name):
            with timer('Insert data to ElasticSearch'):
                mapping = self.load_config()
                self.es.indices.create(index=index_name, body=mapping, ignore=400)
                docs = []
                for i, doc in enumerate(self.contexts):
                    docs.append({
                        '_id':i,
                        '_index':self.index_name,
                        '_source':{'document_text':doc}
                    })
                helpers.bulk(self.es, docs)

    def load_config(self):
        index_config = {"settings": {}, "mappings": {}}
        index_config = self.update_settings(index_config)
        index_config = self.update_mappings(index_config)
        return index_config

    def update_settings(self, index_config):
        """ Update index config's `settings` contents using DataArguments """

        _analyzer = {
            "my_analyzer": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": self.args.decompound_mode,
            }
        }
        if self.args.use_korean_stopwords:
            _analyzer["my_analyzer"].update({"stopwords": "_korean_"})
        if self.args.use_korean_synonyms:
            _analyzer["my_analyzer"].update({"synonyms": "_korean_"})

        _filter = {}
        es_filter = []
        # 영문 소문자 변환
        if self.args.lowercase:
            es_filter += ["lowercase"]
        # 사용자 설정 stopword
        if self.args.stopword_path:
            stop_filter = {
                "type": "stop",
                "stop_words_path": self.args.stopword_path
            }
            _filter.update({"stop_filter": stop_filter})
            es_filter += ["stop_filter"]
        # 한자 음독 변환
        if self.args.nori_readingform:
            es_filter += ["nori_readingform"]
        # chinese-japanese-korean bigram
        if self.args.cjk_bigram:
            es_filter += ["cjk_bigram"]
        # 아라비아 숫자 외에 문자를 전부 아라비아 숫자로 변경
        if self.args.decimal_digit:
            es_filter += ["decimal_digit"]
        _analyzer["my_analyzer"].update({"filter": es_filter})

        all_sim_kwargs = {
            "bm25_similarity": {
                'type': 'BM25',
                'b': self.args.b,
                'k1': self.args.k1,
            },
            "dfr_similarity": {
                'type': 'DFR',
                "basic_model": self.args.dfr_basic_model,
                "after_effect": self.args.dfr_after_effect,
                "normalization": self.args.es_normalization,
            },
            "dfi_similarity": {
                'type': 'DFI',
                "independence_measure": self.args.dfi_measure,
            },
            "ib_similarity": {
                'type': 'IB',
                "distribution": self.args.ib_distribution,
                "lambda": self.args.ib_lambda,
                "normalization": self.args.es_normalization,
            },
            # LM Dirichlet
            "lmd_similarity": {
                'type': 'LMDirichlet',
                "mu": self.args.lmd_mu,
            },
            # LM Jelinek Mercer
            "lmjm_similarity": {
                'type': 'LMJelinekMercer',
                "lambda": self.args.lmjm_lambda,
            },
        }
        sim_kwargs = all_sim_kwargs[self.args.es_similarity]

        assert self.args.es_similarity in all_sim_kwargs.keys()

        index_config["settings"] = {
            "analysis": {"filter": _filter, "analyzer": _analyzer},
            "similarity": {
                self.args.es_similarity: {
                    "type": self.args.es_similarity,
                    **sim_kwargs,
                },
            }
        }
        return index_config

    def update_mappings(self, index_config):
        """ Update index config's `mappings` contents using DataArguments """

        index_config["mappings"] = {
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic.html
            "dynamic": "strict",  # 새 필드가 감지되면 예외가 발생하고 문서가 거부됨.
            "properties": {
                "document_text": {"type": "text", "analyzer": "my_analyzer",
                                  "similarity": self.args.es_similarity}
            }
        }
        return index_config

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search with bm25"):
                doc_scores, doc_indices,doc_contexts = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval with bm25 : ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(doc_contexts[idx])
                    ,
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total) # correct qa
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("searching"):
            responses = self.es.search(index=self.index_name,body=self.mk_query([query],k))["responses"]
        assert (
            len(responses) != 0
        ), "오류가 발생했습니다. 이 오류는 elasticsearch api에서 결과를 받아오지 못할 때 발생합니다."

        doc_scores = [[hit["_score"] for hit in response["hits"]["hits"]] for response in responses]
        doc_indices = [[hit["_id"] for hit in response["hits"]["hits"]] for response in responses]
        return doc_scores[0], doc_indices[0]
    def mk_query(self, query, topk):
        return {"query": {"match": {"document_text": query}}, "size": topk}

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        body = []
        for i in range(len(queries)):
            body.append({"index": self.index_name})
            body.append(self.mk_query(queries[i], k))

        responses = self.es.msearch(body=body)["responses"]

        doc_scores = [[hit["_score"] for hit in response["hits"]["hits"]] for response in responses]
        doc_indices = [[hit["_id"] for hit in response["hits"]["hits"]] for response in responses]
        doc_contexts = [[hit["_source"]["document_text"] for hit in response["hits"]["hits"]] for response in responses]

        return doc_scores, doc_indices, doc_contexts

