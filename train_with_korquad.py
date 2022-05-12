import logging
import sys
from typing import NoReturn
import os
import torch.cuda

from arguments import *
from Data.preprocessing import *

from datasets import DatasetDict, load_from_disk, load_metric, concatenate_datasets, load_dataset
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions, dict2str, str2dict
import wandb

# wandb.init(project="ODQA", entity="bo-lim",run_name='bolim/top_k 10->30')
logger = logging.getLogger(__name__)

def main():
    torch.cuda.empty_cache()
    model_args, data_args, training_args = return_arg()

    """
    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)
    """

    set_seed(training_args.seed)

######################################################################

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    #####################
    training_args.do_train = True
    training_args.do_eval = True
    ####################
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, tokenizer, model)

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    tokenizer,
    model,
) -> NoReturn:

    wandb.init(
        project='MRC',
        entity='miml',
        name='roberta_korquad_noval'
    )

    datasets = load_from_disk(data_args.dataset_name)

    korquad = load_dataset('squad_kor_v1')

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
        column_names_korquad = korquad['train'].column_names
    else:
        column_names = datasets["validation"].column_names
        column_names_korquad = korquad['validation'].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    question_column_name_k = "question" if "question" in column_names_korquad else column_names_korquad[0]
    context_column_name_k = "context" if "context" in column_names_korquad else column_names_korquad[1]
    answer_column_name_k = "answers" if "answers" in column_names_korquad else column_names_korquad[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"



    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    last_checkpoint_k, max_seq_length_k = check_no_error(
        data_args, training_args, korquad, tokenizer
    )
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        origin_train_dataset = datasets["train"]
        train_korquad = korquad['train']
        eval_korquad = korquad['validation']

        # dataset에서 train feature를 생성합니다.
        origin_train_dataset = origin_train_dataset.map(
            function=lambda x: prepare_train_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
                                                      context_column_name=context_column_name, question_column_name=question_column_name,
                                                      answer_column_name=answer_column_name,
                                                      data_args=data_args, max_seq_length=max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        korquad_train_dataset = train_korquad.map(
            function=lambda x: prepare_train_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
                                                      context_column_name=context_column_name_k, question_column_name=question_column_name_k,
                                                      answer_column_name=answer_column_name_k,
                                                      data_args=data_args, max_seq_length=max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names_korquad,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        korquad_eval_dataset = eval_korquad.map(
            function=lambda x: prepare_train_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
                                                      context_column_name=context_column_name_k, question_column_name=question_column_name_k,
                                                      answer_column_name=answer_column_name_k,
                                                      data_args=data_args, max_seq_length=max_seq_length_k),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names_korquad,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        train_dataset = concatenate_datasets(dsets=[origin_train_dataset, korquad_train_dataset, korquad_eval_dataset])
        print(f"TRAIN DATASET: {train_dataset}")

    if training_args.do_eval:
        origin_eval = datasets["validation"]
        #eval_korquad = korquad['validation']

        # Validation Feature 생성
        origin_eval_dataset = origin_eval.map(
            function=lambda x: prepare_validation_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
                                                      context_column_name=context_column_name, question_column_name=question_column_name,
                                                      answer_column_name=answer_column_name,
                                                      data_args=data_args, max_seq_length=max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
        # korquad_eval_dataset = eval_korquad.map(
        #     function=lambda x: prepare_validation_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
        #                                               context_column_name=context_column_name_k, question_column_name=question_column_name_k,
        #                                               answer_column_name=answer_column_name_k,
        #                                               data_args=data_args, max_seq_length=max_seq_length_k),
        #     batched=True,
        #     num_proc=data_args.preprocessing_num_workers,
        #     remove_columns=column_names_korquad,
        #     load_from_cache_file=not data_args.overwrite_cache,
        # )
        #eval_dataset = concatenate_datasets(dsets=[origin_eval_dataset, korquad_eval_dataset])
        eval_dataset = origin_eval_dataset
        eval_example = datasets['validation']
        print(f"EVAL DATASET: {eval_dataset}")

    # eval_example 생성 (korquad+original)

    # origin_eval = origin_eval.map(dict2str)
    # eval_korquad = eval_korquad.map(dict2str)

    # origin_eval = origin_eval.remove_columns(['__index_level_0__', 'document_id'])
    # origin_eval = origin_eval.cast(eval_korquad.features)
    # eval_example = concatenate_datasets([eval_korquad, origin_eval])
    # eval_example = eval_example.map(str2dict)

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in eval_example
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_example if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
