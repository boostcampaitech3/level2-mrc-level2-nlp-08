from BM25Retrieval import BM25_PLUS
from DenseRetrieval import DenseRetrieval
from hybrid import HybridRetrieval
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
import time
from contextlib import contextmanager
import encoder


# tokenizer 정의
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenize_fn = tokenizer.tokenize


# bm25 - wiki_doc
bm = BM25_PLUS(tokenize_fn)
bm.get_sparse_embedding()  # "/opt/ml/input/data/bm25.bin"


# bm25 - train dataset (dense train에 사용)
tdf = bm.retrieve(mrc_train_dataset, topk=10)
train_path = "/opt/ml/input/data/bm25_train.bin"
with open(train_path, "wb") as f:
    pickle.dump(tdf, f)
print("Load BM25_train pickle")


# bm25 - valid dataset (dense train에 사용)
vdf = bm.retrieve(mrc_valid_dataset, topk=10)
train_path = "/opt/ml/input/data/bm25_valid.bin"
with open(train_path, "wb") as f:
    pickle.dump(vdf, f)
print("Load BM25_valid pickle")


# passage encoder & question encoder train
args = encoder.make_easydict()
p_encoder, q_encoder = encoder.train(args=args)  # /opt/ml/input/code/p_encoder, /opt/ml/input/code/q_encoder 에 각각 저장되어 있음


# dense retrieval 학습
p_encoder_path = "/opt/ml/input/code/p_encoder"
q_encoder_path = "/opt/ml/input/code/q_encoder"


### 이후 inference.py 파일 실행하여 결과 파일 도출(default : hybrid ver(bm25 + dense))