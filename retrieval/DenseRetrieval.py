import json
import os
import pickle
import time
from typing import List, NoReturn, Optional, Tuple, Union
import faiss
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi, BM25Plus
import time
from contextlib import contextmanager
from RetrievalBase import Base
from BM25Retrieval import BM25_PLUS
from utils_retrieval import *

class DenseRetrieval(Base):
    def __init__(
        self,
        tokenizer,
        p_encoder_path,
        q_encoder_path,
        data_path="/opt/ml/input/data/",
        caching_path="caching/",
        context_path="wiki_documents.json",
    ):
        super().__init__(
            tokenizer,
            data_path=data_path,
            caching_path=caching_path,
            context_path=context_path,
        )
        print(data_path, caching_path, context_path)
        self.wiki_text = self.wiki_dataset["text"]
        self.wiki_id = self.wiki_dataset["document_id"]
        self.wiki_title = self.wiki_dataset["title"]
        self.data_path = data_path
        self.q_encoder = RobertaEncoder.from_pretrained(q_encoder_path)
        self.p_encoder = RobertaEncoder.from_pretrained(p_encoder_path)
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()

        dense_embedding_path = data_path + caching_path + "dense_embedding.bin"

        if os.path.isfile(dense_embedding_path):
            with open(dense_embedding_path, "rb") as f:
                self.p_embedding = pickle.load(f)
        else:
            self.p_embedding = self.get_dense_embedding(self.p_encoder)
            with open(dense_embedding_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

    def get_dense_embedding(self, p_encoder):
        eval_batch_size = 32

        p_seqs = self.tokenizer(
            self.wiki_corpus,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"]
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)

        p_embedding = []

        with torch.no_grad():

            epoch_iterator = tqdm(dataloader, desc="Iteration", position=0, leave=True)
            p_encoder.eval()
            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = p_encoder(**p_inputs).to("cpu").numpy()
                p_embedding.extend(outputs)
        torch.cuda.empty_cache()
        p_embedding = np.array(p_embedding)
        return p_embedding

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

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        print(isinstance(query_or_dataset, Dataset))
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.wiki_text[doc_indices[i]])

            return (doc_scores, [self.wiki_text[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            print("yes")
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval : ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx], # 리스트 아닌가?
                    "context": " ".join(
                        [self.wiki_text[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total) # correct qa
            return cqas
    
    
    
    def get_relevant_doc(self, query, topk):
        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)

            p_embedding = torch.Tensor(self.p_embedding).squeeze()  # (num_passage, emb_dim)
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embedding, 0, 1))

            rank = (
                torch.argsort(dot_prod_scores, dim=1, descending=True)
                .squeeze()
                .to("cpu")
                .numpy()
                .tolist()
            )
            scores = []
            for r in rank[:topk]:
                scores.append(dot_prod_scores[0][r].item())

        return rank[:top_k], scores

    def get_relevant_doc_bulk(self, querys, topk):
        q_seqs = self.tokenizer(
            querys,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )
        query_sampler = SequentialSampler(dataset)
        query_dataloader = DataLoader(dataset, sampler=query_sampler, batch_size=32)
        q_embs = []
        with torch.no_grad():

            epoch_iterator = tqdm(
                query_dataloader, desc="Iteration", position=0, leave=True
            )
            self.q_encoder.eval()

            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)

                q_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                outputs = self.q_encoder(**q_inputs).to("cpu").numpy()
                q_embs.extend(outputs)
        q_embs = np.array(q_embs)
        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(self.p_embedding).to("cuda")
            q_embs_cuda = torch.Tensor(q_embs).to("cuda")
        dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        doc_score, doc_indices = [], []
        for i in tqdm(range(len(querys))):
            p_list = []
            scores = []
            for j in range(top_k):
                p_list.append(self.context2id_dict[self.wiki_corpus[rank[i][j]]])
                scores.append(dot_prod_scores[i][rank[i][j]].item())
            doc_indices.append(p_list)
            doc_score.append(scores)

        return doc_score, doc_indices