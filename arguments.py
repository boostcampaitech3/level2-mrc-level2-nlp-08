from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser
import yaml

def return_arg():
    # parser = HfArgumentParser(
    #     (ModelArguments, DataTrainingArguments, MyTrainArguments)
    # )

    # [model_args, data_args, training_args,_] = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    with open('configs_bolim/training_args.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    training_arguments, model_arguments, data_arguments = configs['TrainingArguments'], configs['ModelArguments'], configs['DataTrainingArguments']
    model_args = ModelArguments(**model_arguments)
    data_args = DataTrainingArguments(**data_arguments)
    training_args = TrainingArguments(**training_arguments)

    return model_args, data_args, training_args


def return_arg4ES():
    with open('configs_bolim/retrieval_args.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    es_arguments = configs['ElasticSearchArguments']
    es_args = ElasticSearchArguments(**es_arguments)
    return es_args

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

@dataclass
class MyTrainArguments(TrainingArguments):

    output_dir: str = field(
        default='./results',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
@dataclass
class ElasticSearchArguments:
    index_name: str = field(
        default="wiki-index",
        metadata={"help": "The name of the index to use in Elasticsearch."}
    )
    stopword_path: str = field(
        default="user_dic/my_stop_dic.txt",
        metadata={"help": "Path of stopword to use in Elasticsearch."}
    )
    decompound_mode: str = field(
        default="mixed",
        metadata={
            "help": "Determines how the tokenizer handles compound tokens."
            "https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori-tokenizer.html"
        }
    )
    b: float = field(
        default=0.75,
        metadata={
            "help": "Controls to what degree document length normalizes tf values. "
            "The default value is 0.75"
        }
    )
    k1: float = field(
        default=1.2,
        metadata={
            "help": "Controls non-linear term frequency normalization (saturation). "
            "The default value is 1.2."
        }
    )
    es_host_address: str = field(
        default="localhost:9200",
        metadata={"help": "Network host address"}
    )
    es_timeout: int = field(
        default=30,
        metadata={"help": "Determine connection timeout thershold."}
    )
    es_max_retries: int = field(
        default=10,
        metadata={"help": "Determine connection timeout thershold."}
    )
    es_retry_on_timeout: bool = field(
        default=True,
        metadata={"help": "Specifies the maximum number of connection attempts."}
    )
    es_similarity: str = field(
        default="bm25_similarity",
        metadata={"help": "Decide which similarity calculation to use."}
    )
    use_korean_stopwords: bool = field(
        default=False,
        metadata={
            "help": "Decide whether to use the Korean stopword dictionary provided by Elastic Search."}
    )
    use_korean_synonyms: bool = field(
        default=False,
        metadata={
            "help": "Decide whether to use the Korean synonym dictionary provided by Elastic Search."}
    )
    lowercase: bool = field(
        default=False,
        metadata={"help": "Determines whether text is treated as lowercase."}
    )
    nori_readingform: bool = field(
        default=False,
        metadata={
            "help": "Filter rewrites tokens written in Hanja to their Hangul form."}
    )
    cjk_bigram: bool = field(
        default=False,
        metadata={
            "help": "Determines whether to use bigram for Chinese, English and Korean."}
    )
    decimal_digit: bool = field(
        default=False,
        metadata={"help": "Filter folds unicode digits to 0-9"}
    )
    dfr_basic_model: str = field(
        default="g",
        metadata={
            "help": "Basic model of information content for `divergence from randomness`.",
            "choices": ["g", "if", "in", "ine"],
        }
    )
    dfr_after_effect: str = field(
        default="l",
        metadata={
            "help": "First normalization of information gain.",
            "choices": ["b", "l"],
        }
    )
    es_normalization: str = field(
        default="h2",
        metadata={
            "help": "Second (length) normalization",
            "choices": ["no", "h1", "h2", "h3", "z"],
        }
    )
    dfi_measure: str = field(
        default="standardized",
        metadata={
            "help": "Three basic measures of divergence from independence",
            "choices": ["standardized", "saturated", "chisquared"],
        },
    )
    ib_distribution: str = field(
        default="ll",
        metadata={
            "help": "Probabilistic distribution used to model term occurrence",
            "choices": ["ll", "spl"],
        }
    )
    ib_lambda: str = field(
        default="df",
        metadata={
            "help": ":math:`λ_w` parameter of the probability distribution",
            "choices": ["df", "ttf"],
        }
    )
    lmd_mu: int = field(
        default=2000,
        metadata={"help": "Parameters to be used in `LM Dirichlet similarity`."}
    )
    lmjm_lambda: float = field(
        default=0.1,
        metadata={
            "help": (
                "The optimal value depends on both the collection and the query. "
                "The optimal value is around 0.1 for title queries and 0.7 for long queries. "
                "Default to 0.1. When value approaches 0, "
                "documents that match more query terms will be ranked higher than those that match fewer terms."
            )
        }
    )