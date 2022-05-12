import argparse

from numpy.lib.function_base import gradient
from utils_retrieval import *
from transformers import (
    AutoTokenizer,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
import random
import numpy as np

def to_cuda(batch):
    return tuple(t.cuda() for t in batch)


def train_with_negative(
    args, p_encoder, q_encoder, train_dataset, valid_dataset, num_neg
):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        # question
        {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    best_loss = 9999  # valid_loss를 저장하는 변수
    best_acc = -1 # acc를 저장하는 변수
    num_epoch = 0 

    for _ in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        train_loss = 0
        train_acc = 0
        train_step = 0
        for batch in epoch_iterator:
            train_step += 1
            q_encoder.train()
            p_encoder.train()

            neg_batch_ids = [] # negative batch input_ids
            neg_batch_att = [] # negative batch attention_mask
            neg_batch_tti = [] # negative batch token_type_ids
            random_sampling_idx = random.randrange(0, num_neg)
            for batch_in_sample_idx in range(args.per_device_train_batch_size):
                '''
                question과 pos passage는 1대1로 매칭이 되지만
                hard negative sample들은 해당 question에 대해 num_neg의 수만큼 매칭이 되기 때문에
                매 학습 루프마다 한개를 랜덤하게 뽑아서 pos passage와 concat을 하여 사용하게 됩니다.
                '''
                neg_batch_ids.append(
                    batch[3][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0)
                )
                neg_batch_att.append(
                    batch[4][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0)
                )
                neg_batch_tti.append(
                    batch[5][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0)
                )
            neg_batch_ids = torch.cat(neg_batch_ids)
            neg_batch_att = torch.cat(neg_batch_att)
            neg_batch_tti = torch.cat(neg_batch_tti)
            p_inputs = {
                "input_ids": torch.cat((batch[0], neg_batch_ids), 0).cuda(),
                "attention_mask": torch.cat((batch[1], neg_batch_att), 0).cuda(),
                "token_type_ids": torch.cat((batch[2], neg_batch_tti), 0).cuda(),
            }

            q_inputs = {
                "input_ids": batch[6].cuda(),
                "attention_mask": batch[7].cuda(),
                "token_type_ids": batch[8].cuda(),
            }

            p_outputs = p_encoder(**p_inputs)  # (batch_size * 2, emb_dim)
            q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size * 2) = (batch_size, batch_size * 2)

            # 정답은 대각선의 성분들 -> 0 1 2 ... batch_size - 1
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)
            train_loss += loss.item()

            _, preds = torch.max(sim_scores, 1)  
            train_acc += (
                torch.sum(preds.cpu() == targets.cpu())
                / args.per_device_train_batch_size
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_encoder.zero_grad()
            p_encoder.zero_grad()
            global_step += 1
            # validation
            if train_step % 40 == 0:
                valid_loss = 0
                valid_acc = 0
                v_epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
                for step, batch in enumerate(v_epoch_iterator):
                    with torch.no_grad():
                        q_encoder.eval()
                        p_encoder.eval()

                        cur_batch_size = batch[0].size()[0]  
                        # 마지막 배치의 drop last를 안하기 때문에 단순 batch_size를 사용하면 에러발생
                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)
                        p_inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                        }

                        q_inputs = {
                            "input_ids": batch[6],
                            "attention_mask": batch[7],
                            "token_type_ids": batch[8],
                        }
                        p_outputs = p_encoder(**p_inputs)
                        q_outputs = q_encoder(**q_inputs)

                        sim_scores = torch.matmul(
                            q_outputs, torch.transpose(p_outputs, 0, 1)
                        )
                        targets = torch.arange(0, cur_batch_size).long()
                        if torch.cuda.is_available():
                            targets = targets.to("cuda")

                        sim_scores = F.log_softmax(sim_scores, dim=1)
                        loss = F.nll_loss(sim_scores, targets)

                        _, preds = torch.max(sim_scores, 1)  #
                        valid_acc += (
                            torch.sum(preds.cpu() == targets.cpu()) / cur_batch_size
                        )

                        valid_loss += loss
                valid_loss = valid_loss / len(valid_dataloader)
                valid_acc = valid_acc / len(valid_dataloader)

                print()
                print(f"valid loss: {valid_loss}")
                print(f"valid acc: {valid_acc}")
                if best_loss > valid_loss:
                    # valid_loss가 작아질 때만 저장하고 best_loss와 best_acc를 업데이트
                    # acc에 대해서도 가능합니다.
                    print("best model save")
                    p_encoder.save_pretrained(args.output_dir + "/p_encoder")
                    q_encoder.save_pretrained(args.output_dir + "/q_encoder")
                    best_acc = valid_acc
                    best_loss = valid_loss

        num_epoch += 1
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        print(f"train loss: {train_loss}")
        print(f"train acc: {train_acc}")

    return p_encoder, q_encoder


def train(args):
    print(args)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    train_dataset = prepare_in_batch_negative(
        data_path=args.train_data_path,
        bm25_path=args.train_bm25_path,
        max_context_seq_length=args.max_context_seq_length,
        max_question_seq_length=args.max_question_seq_length,
        neg_num=args.num_neg,
        tokenizer=tokenizer,
    )
    valid_dataset = prepare_in_batch_negative(
        data_path=args.valid_data_path,
        bm25_path=args.valid_bm25_path,
        max_context_seq_length=args.max_context_seq_length,
        max_question_seq_length=args.max_question_seq_length,
        neg_num=args.num_neg,
        tokenizer=tokenizer,
    )

    # p_encoder = BertEncoder.from_pretrained(args.model_checkpoint)
    # q_encoder = BertEncoder.from_pretrained(args.model_checkpoint)
    p_encoder = RobertaEncoder.from_pretrained(args.model_checkpoint)
    q_encoder = RobertaEncoder.from_pretrained(args.model_checkpoint)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    return train_with_negative(training_args, p_encoder, q_encoder, train_dataset, valid_dataset, args.num_neg)

def make_easydict():
    import easydict

    args = easydict.EasyDict({
            "seed": 42,
            "model_checkpoint": "klue/roberta-base",
            "train_data_path": "/opt/ml/input/data/train_dataset/train",
            "valid_data_path": "/opt/ml/input/data/train_dataset/validation",
            "train_bm25_path": "/opt/ml/input/data/bm25_train.bin",
            "valid_bm25_path": "/opt/ml/input/data/bm25_valid.bin",
            "max_context_seq_length": 512,
            "max_question_seq_length": 64,
            "output_dir": "/opt/ml/input/code",
            "evaluation_strategy": "epoch",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 5,
            "weight_decay": 0.01,
            "num_neg": 10
    })
    
    return args