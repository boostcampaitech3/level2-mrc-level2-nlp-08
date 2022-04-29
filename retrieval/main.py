import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
import argparse, yaml
import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils import timer, load_dataset_from_disk
from SparseRetrieval import SparseRetrieval
from FaissRetrieval import FaissRetrieval

def main(args):
    # Test sparse
    
    # 1. load original dataset from disk
    full_ds = load_dataset_from_disk(args.dataset_name)
    
    
    # 2. print full query dataset
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    
    
    # 3. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)


    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    
    # 4. retrieval
    if args.use_faiss:
        retriever = FaissRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
        )

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))
    
    else:
        retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
        )
        
        # test single query
        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            
            print("correct retrieval result by exhaustive search", df["correct"].sum() / len(df))


if __name__ == "__main__":
    with open('./configs/retrieval_args.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**configs)

    main(args)