from transformers import (
    BertModel, BertPreTrainedModel,
    RobertaModel, PreTrainedModel, RobertaConfig
)
import time
from datasets import Dataset, concatenate_datasets, load_from_disk
import torch, random, pickle
from contextlib import contextmanager
import random
import datasets
import torch
from torch.utils.data import Dataset
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# timer
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# 1. load original dataset from disk
def load_dataset_from_disk(dataset_name):    
    org_dataset = load_from_disk(dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    
    return full_ds
    
# check cuda
def to_cuda(batch):
        return tuple(t.cuda() for t in batch)

    
# encoder
class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

# encoder
class RobertaEncoder(RobertaModel):

    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)
        
        self.roberta = RobertaModel(config) # PretrainedConfig
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

    
# seed 고정
def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
       

# prepare_in_batch_negative
class prepare_in_batch_negative(Dataset):
    '''
    dense retrieval 모델을 학습시킬 데이터 셋
    '''
    def __init__(
        self,
        data_path: str,
        bm25_path: str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        neg_num,
        tokenizer,
    ):
        preprocess_data = self.preprocess_pos_neg(
            data_path,
            bm25_path,
            max_context_seq_length,
            max_question_seq_length,
            neg_num,
            tokenizer,
        )
        '''
        data_path
        -> query - context로 이루어진 dataset의 경로
        
        bm25_path
        -> query에 대해서 bm25로 찾아낸 유사도가 높은 context 데이터
        retrieval/SparseRetrieval의 get_topk_doc_id_and_score_for_querys 메서드로 
        해당 bin 파일을 만들 수 있습니다.
        max_context_seq_length
        -> context의 max_length
        max_question_seq_length
        -> question의 max_lenth
        neg_num
        -> retrieval 학습에 사용할 Inbatch negative 외에 hard negative sample의 갯수
        '''

        self.p_input_ids = preprocess_data[0]
        self.p_attension_mask = preprocess_data[1]
        self.p_token_type_ids = preprocess_data[2]

        self.np_input_ids = preprocess_data[3]
        self.np_attension_mask = preprocess_data[4]
        self.np_token_type_ids = preprocess_data[5]

        self.q_input_ids = preprocess_data[6]
        self.q_attension_mask = preprocess_data[7]
        self.q_token_type_ids = preprocess_data[8]

    def __len__(self):
        return self.p_input_ids.size()[0]

    def __getitem__(self, index):
        return (
            self.p_input_ids[index],
            self.p_attension_mask[index],
            self.p_token_type_ids[index],
            self.np_input_ids[index],
            self.np_attension_mask[index],
            self.np_token_type_ids[index],
            self.q_input_ids[index],
            self.q_attension_mask[index],
            self.q_token_type_ids[index],
        )

    def preprocess_pos_neg(
        self,
        data_path: str,
        bm25_path: str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        num_neg,
        tokenizer,
    ):
        base_path = "/opt/ml/input/data/"
        caching_path = "caching/"
        caching_context2id_path = base_path + caching_path + "context2id.bin"
        caching_id2context_path = base_path + caching_path + "id2context.bin"
        caching_id2title_path = base_path + caching_path + "id2title.bin"

        # doc_id - context dict
        with open(caching_id2context_path, "rb") as f:
            id2context = pickle.load(f)
        # doc_id - title dict
        with open(caching_id2title_path, "rb") as f:
            id2title = pickle.load(f)
        # context - doc_id dict
        with open(caching_context2id_path, "rb") as f:
            context2id = pickle.load(f)

        # question - doc_id_list
        with open(bm25_path, "rb") as file:  # query - bm25_doc_id
            bm25 = pickle.load(file)
            
        dataset = load_from_disk(data_path)
        dataset = dataset.to_pandas()
        pos_ctx = dataset["context"].to_list()
        pos_title = dataset["title"].to_list()
        questions = dataset["question"].to_list()

        # print(len(pos_ctx))

        neg_ctx = []
        neg_title = []
        for i in tqdm(range(len(pos_ctx))):
            # q = questions[i]  # i 번째 question
            ground_truth = pos_ctx[i]  # 정답 문장
            ground_truth_title = pos_title[i]  # 정답 타이틀
            cnt = num_neg  # 추가할 negative context 갯수
            answer = dataset["answers"][i]["text"][0]  # 정답
            idx = 0

            while cnt != 0:
                # 실습 파일 -> 완전 랜덤
                # 해당 파일 -> 가장 정답에 가까운 오답을 negative로 => bm25 먼저 계산 필요
                neg_ctx_sample = id2context[bm25.iloc[i]["context_id"][idx]] # bm25[i]["context_id"]
                # neg_ctx_sample = id2context[bm25_question_ids[q][idx]]
                if (ground_truth != neg_ctx_sample) and (not answer in neg_ctx_sample):
                    # 비슷한 context를 추가하되 정답을 포함하지 않는 문장을 추가한다.
                    neg_ctx.append(id2context[int(bm25.iloc[i]["context_id"][idx])])
                    neg_title.append(id2title[int(bm25.iloc[i]["context_id"][idx])])
                    cnt -= 1
                idx += 1
                if idx == len(bm25.iloc[i]["context_id"]):
                    # 예외처리 ex) 정답이 전부 포함되서 추가할 문장이 없을 경우
                    idx_step = 1
                    while cnt != 0:
                        temp_neg = pos_ctx[i - idx_step]  # 끝에서부터 -> 가장 유사도 낮은 것 추가
                        temp_neg_title = pos_title[i - idx_step]
                        # 이전에 추가된 ground truth context를 negative sample로 생성
                        neg_ctx.append(temp_neg)
                        neg_title.append(temp_neg_title)
                        idx_step += 1
                        cnt -= 1

        print(f"pos_context cnt: {len(pos_ctx)}")
        print(f"neg_context cnt: {len(neg_ctx)}")

        print(f"pos * num_neg == neg? : {len(pos_ctx) * num_neg == len(neg_ctx)}")

        q_seqs = tokenizer(
            questions,
            max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        p_seqs = tokenizer(
            pos_ctx,
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        np_seqs = tokenizer(
            neg_ctx,
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        max_len = np_seqs["input_ids"].size(-1)
        np_seqs["input_ids"] = np_seqs["input_ids"].view(-1, num_neg, max_len)
        np_seqs["attention_mask"] = np_seqs["attention_mask"].view(-1, num_neg, max_len)
        np_seqs["token_type_ids"] = np_seqs["token_type_ids"].view(-1, num_neg, max_len)

        return (
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            np_seqs["input_ids"],
            np_seqs["attention_mask"],
            np_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )