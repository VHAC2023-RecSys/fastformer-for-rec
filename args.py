import argparse
from typing import List, Optional
from utility import utils
import logging

from tap import Tap


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--root_data_dir",
#         type=str,
#         default="/ads-nfs/t-shxiao/cache/data/Mind_large/",
#     )

#     parser.add_argument("--filename_pat", type=str, default="ProtoBuf_*.tsv")
#     parser.add_argument("--model_dir", type=str, default="./saved_models/")
#     parser.add_argument("--npratio", type=int, default=1)
#     parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
#     parser.add_argument("--enable_shuffle", type=utils.str2bool, default=True)
#     parser.add_argument("--enable_prefetch", type=utils.str2bool, default=True)
#     parser.add_argument("--num_workers", type=int, default=2)
#     parser.add_argument("--log_steps", type=int, default=200)
#     parser.add_argument("--batch_size", type=int, default=64)

#     # model training
#     parser.add_argument("--epochs", type=int, default=6)
#     parser.add_argument("--lr", type=float, default=0.0001)
#     parser.add_argument(
#         "--news_attributes",
#         type=str,
#         nargs="+",
#         default=["title", "abstract"],
#         choices=["title", "abstract", "body", "category", "domain", "subcategory"],
#     )

#     parser.add_argument("--num_words_title", type=int, default=32)
#     parser.add_argument("--num_words_abstract", type=int, default=50)
#     parser.add_argument("--num_words_body", type=int, default=100)

#     parser.add_argument("--user_log_length", type=int, default=100)

#     parser.add_argument(
#         "--word_embedding_dim",
#         type=int,
#         default=300,
#     )
#     parser.add_argument("--news_dim", type=int, default=64)
#     parser.add_argument("--demo_dim", type=int, default=64)
#     parser.add_argument(
#         "--news_query_vector_dim",
#         type=int,
#         default=200,
#     )
#     parser.add_argument(
#         "--user_query_vector_dim",
#         type=int,
#         default=32,
#     )
#     parser.add_argument(
#         "--num_attention_heads",
#         type=int,
#         default=20,
#     )
#     parser.add_argument(
#         "--attention_dims",
#         type=int,
#         default=20,
#     )
#     parser.add_argument("--user_log_mask", type=utils.str2bool, default=True)
#     parser.add_argument("--drop_rate", type=float, default=0.2)
#     parser.add_argument("--save_steps", type=int, default=100000)
#     parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

#     parser.add_argument(
#         "--load_ckpt_name",
#         type=str,
#         default=None,
#         help="choose which ckpt to load and test",
#     )
#     # share
#     parser.add_argument("--title_share_encoder", type=utils.str2bool, default=False)

#     # Turing
#     parser.add_argument(
#         "--pretrained_model", type=str, default="unilm", choices=["unilm", "others"]
#     )
#     parser.add_argument("--pretrained_model_path", type=str, default="../tnlr")
#     parser.add_argument(
#         "--config-name", type=str, default="unilm2-base-uncased-config.json"
#     )
#     parser.add_argument(
#         "--model_name_or_path", type=str, default="unilm2-base-uncased.bin"
#     )
#     parser.add_argument(
#         "--tokenizer_name", type=str, default="unilm2-base-uncased-vocab.txt"
#     )

#     parser.add_argument("--num_hidden_layers", type=int, default=-1)

#     parser.add_argument(
#         "--use_pretrain_news_encoder", type=utils.str2bool, default=False
#     )
#     parser.add_argument(
#         "--freeze_pretrain_news_encoder", type=utils.str2bool, default=False
#     )

#     # new parameters for speedyrec
#     parser.add_argument("--warmup", type=utils.str2bool, default=False)
#     parser.add_argument("--world_size", type=int, default=-1)
#     parser.add_argument("--enable_prefetch_stream", type=utils.str2bool, default=True)
#     parser.add_argument("--pretrain_lr", type=float, default=1e-4)
#     parser.add_argument(
#         "--beta_for_cache",
#         type=float,
#         help="the hyper parameter for the growth rate of lookup  probability",
#         default=0.002,
#     )
#     parser.add_argument("--max_step_in_cache", type=int, help="\gamma", default=20)
#     parser.add_argument("--savename", type=str, default="speedy")
#     parser.add_argument("--warmup_step", type=int, default=2000)
#     parser.add_argument("--schedule_step", type=int, default=30000)
#     parser.add_argument("--test_steps", type=int, default=1000000)
#     parser.add_argument("--max_hit_ratio", type=float, default=1)

#     args = parser.parse_args()
#     logging.info(args)
#     return args


class Args(Tap):
    root_data_dir: Optional[str] = "./data/speedy_data"
    filename_pat: Optional[str] = "ProtoBuf_*.tsv"
    model_dir: Optional[str] = "./saved_models/"
    npratio: Optional[int] = 1
    enable_gpu: Optional[bool] = True
    enable_shuffle: Optional[bool] = True
    enable_prefetch: Optional[bool] = True
    num_workers: Optional[int] = 2
    log_steps: Optional[int] = 200
    batch_size: Optional[int] = 64

    # model training
    epochs: Optional[int] = 6
    lr: Optional[float] = 1e-4
    news_attributes: Optional[List[str]] = [
        "title",
        "abstract",
    ]  # choices=["title", "abstract", "body", "category", "domain", "subcategory"]
    num_words_title: Optional[int] = 32
    num_words_abstract: Optional[int] = 50
    num_words_body: Optional[int] = 100
    user_log_length: Optional[int] = 100
    word_embedding_dim: Optional[int] = 300
    news_dim: Optional[int] = 64
    demo_dim: Optional[int] = 64
    news_query_vector_dim: Optional[int] = 200
    user_query_vector_dim: Optional[int] = 32
    num_attention_heads: Optional[int] = 20
    attention_dims: Optional[int] = 20
    user_log_mask: Optional[bool] = True
    drop_rate: Optional[float] = 0.2
    save_steps: Optional[int] = 1e5
    max_steps_per_epoch: Optional[int] = 1e6
    load_ckpt_name: Optional[str] = None  # "choose which ckpt to load and test"

    # share
    title_share_encoder: Optional[bool] = False

    # turing
    pretrained_model: Optional[str] = "unilm"  # choices=["unilm", "others"]
    pretrained_model_path: Optional[str] = "../tnlr"
    # config_name: Optional[str] = "unilm2-base-uncased-config.json"
    # model_name_or_path: Optional[str] = "unilm2-base-uncased.bin"
    # tokenizer_name: Optional[str] = "unilm2-base-uncased-vocab.txt"

    num_hidden_layers: Optional[int] = -1
    use_pretrain_news_encoder: Optional[bool] = False
    freeze_pretrain_news_encoder: Optional[bool] = False

    warmup: Optional[bool] = False
    world_size: Optional[int] = -1
    enable_prefetch_stream: Optional[bool] = True
    pretrain_lr: Optional[float] = 1e-4
    beta_for_cache: Optional[
        float
    ] = 2e-3  # "the hyper parameter for the growth rate of lookup  probability"
    max_step_in_cache: Optional[int] = 20
    savename: Optional[str] = "speedy"
    warmup_step: Optional[int] = 2000
    schedule_step: Optional[int] = 3e4
    test_steps: Optional[int] = 1e6
    max_hit_ratio: Optional[float] = 1


def parse_args():
    args = Args().parse_args()
    logging.info(args)
    return args
