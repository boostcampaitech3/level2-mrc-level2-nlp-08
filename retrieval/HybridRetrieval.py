import json
import os
import pickle
import time
from typing import List, NoReturn, Optional, Tuple, Union
import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi, BM25Plus
import time, torch
from contextlib import contextmanager
from datasets import Dataset, concatenate_datasets, load_from_disk
from RetrievalBase import Base
from BM25Retrieval import BM25_PLUS
from DenseRetrieval import DenseRetrieval

class HybridRetrieval(Base):
    def __init__(
        self,
        tokenizer,
        q_encoder_path,
        p_encoder_path,
        data_path="/opt/ml/input/data/",
        caching_path="caching/",
        context_path="wikipedia_documents.json",
    ):
        super().__init__(
            tokenizer,
            data_path=data_path,
            caching_path=caching_path,
            context_path=context_path,
        )
        self.wiki_text = self.wiki_dataset["text"]
        self.wiki_id = self.wiki_dataset["document_id"]
        self.wiki_title = self.wiki_dataset["title"]
        self.data_path = data_path
        
        self.sparse_retrieval = BM25_PLUS(tokenizer=tokenizer, tokenize_fn=tokenizer.tokenize)
        self.dense_retrieval = DenseRetrieval(
            tokenizer=tokenizer,
            p_encoder_path=p_encoder_path,
            q_encoder_path=q_encoder_path,
        )
        self.q_encoder = self.dense_retrieval.q_encoder
        self.p_embedding = self.dense_retrieval.p_embedding
        if torch.cuda.is_available():
            self.p_embedding = torch.Tensor(self.p_embedding).to("cuda")

    def get_relevant_doc(self, query, topk):
        doc_score, doc_indices = self.sparse_retrieval.get_relevant_doc(
            query=query, k=topk
        )
        return self.rerank(query, doc_score, doc_indices)

    def get_relevant_doc_bulk(self, querys, topk):
        doc_score, doc_indices = {}, {}

        for i in tqdm(range(len(querys))):
            query = querys[i]
            score, indices = self.get_relevant_doc(query, topk)
            doc_indices[query] = indices
            doc_score[query] = score

        return doc_score, doc_indices
    
    def rerank(self, query, score, indices):
        p_embedding = self.p_embedding
        with torch.no_grad():
            self.q_encoder.eval()
            q_seqs_val = self.tokenizer(
                query, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cuda")
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        rank = rank.cpu().numpy().tolist()

        bm25_id2score = {k: v for k, v in zip(indices, score)}

        hybrid_id2score = dict()

        for i in range(len(rank)):
            dense_id = self.context2id_dict[self.wiki_corpus[i]]
            if dense_id in list(bm25_id2score.keys()):
                hybrid_id2score[dense_id] = dot_prod_scores[0][i].item() + bm25_id2score[dense_id]  # 덧셈으로 rerank

        hybrid_id2score = list(hybrid_id2score.items())
        hybrid_id2score.sort(key=lambda x: x[1], reverse=True)
        doc_indices = list(map(lambda x: x[0], hybrid_id2score))
        doc_score = list(map(lambda x: x[1], hybrid_id2score))

        return doc_score, doc_indices