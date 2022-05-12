import pickle
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import numpy as np

class Base:
    def __init__(
        self,
        tokenizer,
        data_path="/opt/ml/input/data/",
        caching_path="caching/",
        context_path="wikipedia_documents.json",
    ):
        '''
        Retrieval의 최상위 클래스
        Sparse, Dense, Hybrid 모두 이 클래스를 상속받아서 사용합니다.        
        '''
        self.tokenizer = tokenizer
        self.wiki_dataset = pd.read_json("/opt/ml/input/data/wikipedia_documents.json", orient="index")
        caching_context2id_path = data_path + caching_path + "context2id.bin"
        caching_id2context_path = data_path + caching_path + "id2context.bin"
        caching_id2title_path = data_path + caching_path + "id2title.bin"

        if (
            os.path.isfile(caching_context2id_path)
            and os.path.isfile(caching_id2context_path)
            and os.path.isfile(caching_id2title_path)
        ):
            with open(caching_context2id_path, "rb") as f:
                self.context2id_dict = pickle.load(f)
            with open(caching_id2context_path, "rb") as f:
                self.id2context_dict = pickle.load(f)
            with open(caching_id2title_path, "rb") as f:
                self.id2title_dict = pickle.load(f)
        else:
            wiki_text = self.wiki_dataset["text"]
            wiki_id = self.wiki_dataset["document_id"]
            wiki_title = self.wiki_dataset["title"]

            self.context2id_dict = {k: v for k, v in zip(wiki_text, wiki_id)}
            self.id2context_dict = {k: v for k, v in zip(wiki_id, wiki_text)}
            self.id2title_dict = {k: v for k, v in zip(wiki_id, wiki_title)}

            with open(caching_context2id_path, "wb") as file:
                pickle.dump(self.context2id_dict, file)
            with open(caching_id2context_path, "wb") as file:
                pickle.dump(self.id2context_dict, file)
            with open(caching_id2title_path, "wb") as file:
                pickle.dump(self.id2title_dict, file)

        self.wiki_corpus = list(self.context2id_dict.keys())
    
    def get_relevant_doc(self, query, topk):
        pass

    def get_relevant_doc_bulk(self, querys, topk):
        pass