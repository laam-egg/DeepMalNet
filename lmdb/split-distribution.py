"""
The script has the following dependencies:

    pip install lmdb==1.7.3 msgpack==1.1.1 msgpack-numpy==0.4.8 tqdm

or:

    pip install lmdb==1.7.3 msgpack==0.6.2 msgpack-numpy==0.4.8 tqdm

Minimum supported Python version: 3.6

You can also run this script in the
DeepMalNet conda environment.



This script reads the splits (resulted from running
the split.py script in this same directory) and
prints out how many percent of each split are 0-labelled
and 1-labelled records.
"""

import os, sys
import lmdb
from tqdm import tqdm
import numpy as np
import msgpack
import msgpack_numpy
from typing import Iterable
import ast

msgpack_numpy.patch()

def read_label_from_value(raw_value):
    payload = msgpack.unpackb(raw_value, raw=False)
    label = payload['lb']
    # features = payload['ef']
    return label

def print_split_distribution(num_label_0, num_label_1):
    num_total = num_label_0 + num_label_1
    percent_0 = round(num_label_0 / num_total * 100, 2)
    percent_1 = 100 - percent_0
    print(f"Split distribution: {percent_0}% zero, {percent_1}% one")

def walk_split_files(splits_dir_path):
    # type: (str) -> Iterable[str]
    file_names = os.listdir(splits_dir_path)
    for file_name in file_names:
        if not file_name.endswith(".txt"):
            continue
        full_file_path = os.path.join(splits_dir_path, file_name)
        if not os.path.isfile(full_file_path):
            continue
        yield full_file_path

def run(lmdb_path, splits_dir_path):
    # type: (str, str) -> None

    env = lmdb.open(
        lmdb_path,
        map_size=1024 * 1024 * 1024 * 1024, # 1 TB
        readonly=True,
        lock=False,
    )

    with env.begin(write=False) as txn:
        for split_file_path in walk_split_files(splits_dir_path):
            print(f"Analyzing split: {split_file_path}")
            num_0 = num_1 = 0
            with open(split_file_path, 'r') as split_file:
                for line in tqdm(
                    split_file,
                    total=sum(1 for _ in open(split_file_path, 'r')),
                ):
                    line = line.strip().strip(',')
                    key = ast.literal_eval(line)
                    raw_value = txn.get(key)
                    label = read_label_from_value(raw_value)
                    if label == 0:
                        num_0 += 1
                    elif label == 1:
                        num_1 += 1
                    else:
                        raise ValueError(f"Invalid label at key {repr(key)}: {repr(label)}")
            
            print_split_distribution(num_0, num_1)

def main():
    if len(sys.argv) != 3:
        raise RuntimeError(f"Exactly 3 arguments are expected, got {len(sys.argv)} instead")

    print(sys.argv) 
    _argv0, lmdb_path, output_dir_path = sys.argv
    run(lmdb_path, output_dir_path)

if __name__ == "__main__":
    main()
