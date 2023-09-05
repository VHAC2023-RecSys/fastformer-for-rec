# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from tap import Tap
import math
from pathlib import Path
import argparse
import os


def generate_news(raw_data_path: str):
    for mode in ["train", "dev", "test"]:
        Path(f"./data/speedy_data/{mode}").mkdir(parents=True, exist_ok=True)
        fsave = open(
            "./data/speedy_data/{}/docs.tsv".format(mode), "w", encoding="utf-8"
        )
        for line in open(
            os.path.join(raw_data_path, f"{mode}/news.tsv"), encoding="utf-8"
        ):
            (
                doc_id,
                category,
                subcategory,
                title,
                abstract,
                url,
                entities,
                abstract_entites,
            ) = line.strip().split("\t")
            body = " "
            fsave.write(
                doc_id
                + "\t"
                + category
                + "\t"
                + subcategory
                + "\t"
                + title
                + "\t"
                + abstract
                + "\t"
                + body
                + "\n"
            )
        fsave.close()


def generate_traindata(raw_data_path: str, file_num=512):
    data = []
    for line in open(
        os.path.join(raw_data_path, "train/behaviors.tsv"),
        "r",
        encoding="utf-8",
    ):
        impid, uid, time, history, impressions = line.strip().split("\t")
        impressions = impressions.split(" ")
        pos = []
        neg = []
        for item in impressions:
            item = item.split("-")
            if int(item[1]) == 1:
                pos.append(item[0])
            else:
                neg.append(item[0])
        for p in pos:
            case = impid + "\t" + uid + "\t" + history + "\t" + p + "\t" + " ".join(neg)
            data.append((uid, case))

    all_l = len(data)
    data = sorted(data, key=lambda x: x[0])
    data = [x[1] for x in data]

    num_each_file = math.floor(all_l / file_num)
    for i in range(file_num):
        start = i * num_each_file
        end = (i + 1) * (num_each_file)
        if i == file_num - 1:
            end = all_l
        subdata = data[start:end]
        save_data(subdata, file_rank=i, mode="train")


def save_data(data, file_rank, mode):
    rank = ["0"] * (4 - len(str(file_rank)))
    rank = "".join(rank) + str(file_rank)
    with open(
        "./data/speedy_data/{}/ProtoBuf_{}.tsv".format(mode, rank),
        "w",
        encoding="utf-8",
    ) as f:
        for case in data:
            f.write(case + "\n")


def generate_testdata(raw_data_path: str, mode: str):
    Path(f"./data/speedy_data/{mode}").mkdir(parents=True, exist_ok=True)
    with open(
        "./data/speedy_data/{}/ProtoBuf_0.tsv".format(mode), "w", encoding="utf-8"
    ) as f:
        for line in open(
            os.path.join(raw_data_path, "{}/behaviors.tsv".format(mode)),
            "r",
            encoding="utf-8",
        ):
            impid, uid, time, history, impressions = line.strip().split("\t")

            if mode == "dev":
                impressions = impressions.split(" ")
                pos = []
                neg = []
                for item in impressions:
                    item = item.split("-")
                    if int(item[1]) == 1:
                        pos.append(item[0])
                    else:
                        neg.append(item[0])
                case = (
                    impid
                    + "\t"
                    + uid
                    + "\t"
                    + history
                    + "\t"
                    + " ".join(pos)
                    + "\t"
                    + " ".join(neg)
                )
            elif mode == "test":
                case = impid + "\t" + uid + "\t" + history + "\t" + impressions

            f.write(case + "\n")


class Args(Tap):
    raw_data_path: Optional[str] = "./datasets/MINDsmall/"
    file_num_for_traindata: Optional[int] = 512


def main():
    args = Args().parse_args()
    assert os.path.exists(os.path.join(args.raw_data_path, "train"))
    assert os.path.exists(os.path.join(args.raw_data_path, "dev"))

    Path("./data/speedy_data/").mkdir(parents=True, exist_ok=True)
    generate_news(args.raw_data_path)
    generate_traindata(args.raw_data_path, file_num=args.file_num_for_traindata)
    generate_testdata(args.raw_data_path, "dev")
    generate_testdata(args.raw_data_path, "test")


if __name__ == "__main__":
    main()
