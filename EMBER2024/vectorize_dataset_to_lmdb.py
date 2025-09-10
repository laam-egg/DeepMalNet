import os
import sys
import lmdb
import json
import multiprocessing as mp
import numpy as np
import polars as pl
from tqdm import tqdm
import msgpack
import msgpack_numpy
from pathlib import Path
from typing import Iterator
from itertools import islice

msgpack_numpy.patch()

from thrember.features import PEFeatureExtractor


SCRIPT_DIR = os.path.dirname(sys.argv[0])
DATASET_DIR = SCRIPT_DIR + "/../dataset/EMBER2024/"

os.makedirs(DATASET_DIR, exist_ok=True)


def vectorize(raw_features_string, extractor, label_type = "label"):
    # type: (str, PEFeatureExtractor, str) -> tuple[bytes, dict] | tuple[None, None]
    """
    Copied from thrember/models.py
    with modifications.


    Vectorize a single sample of raw features and return
    a dict that could be msgpack'ed to LMDB, along with
    the LMDB row key.

    And only focus on binary classification (malware/benign).
    """
    raw_features = json.loads(raw_features_string)
    raw_id = raw_features["sha256"] # type: str
    id = raw_id.encode('ascii')
    feature_vector = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError("Invalid label_type!")
    label = raw_features[label_type]

    if not isinstance(label, int):
        return None, None
    
    return id, {
        "lb": label,
        "ef": feature_vector,
    }

def worker(lmdb_path, raw_features_string_batch, extractor, label_type = "label"):
    # type: (str, list[str], PEFeatureExtractor, str) -> dict | None
    env = lmdb.open(
        lmdb_path, map_size=1024 * 1024 * 1024 * 1024, # 1 TB
        subdir=True,
        lock=True,
        writemap=False,
        sync=False,
    )

    with env.begin(write=True) as txn:
        for raw_features_string in raw_features_string_batch:
            key, value = vectorize(raw_features_string, extractor, label_type)
            if key is None:
                continue
            txn.put(key, msgpack.packb(value, use_bin_type=True))

def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)

def raw_feature_iterator():
    # type: () -> Iterator[str]
    """
    Copied from thrember/models.py
    with modifications.
    

    Yield raw feature strings from
    EMBER2024 jsonl dataset
    """
    for filename in tqdm(os.listdir(DATASET_DIR)):
        if not filename.endswith(".jsonl"):
            continue
        path = DATASET_DIR + "/" + filename
        with open(path, 'r') as fin:
            for line in fin:
                yield line

def chunked_iterable(iterable, size):
    """Yield successive chunks from an iterable lazily."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def vectorize_args_iterator(lmdb_path, extractor, batch_size):
    # type: (str, PEFeatureExtractor, int) -> Iterator[tuple[str, list[str], PEFeatureExtractor]]
    for batch in chunked_iterable(raw_feature_iterator(), batch_size):
        yield lmdb_path, batch, extractor

def worker_forward_args(args):
    return worker(*args)

def vectorize_dataset_to_lmdb(lmdb_path):
    # type: (str) -> None

    extractor = PEFeatureExtractor()
    BATCH_SIZE = 100

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in pool.imap_unordered(
            worker_forward_args,
            vectorize_args_iterator(lmdb_path, extractor, BATCH_SIZE),
        ):
            pass

def main():
    lmdb_path = sys.argv[1]
    vectorize_dataset_to_lmdb(lmdb_path)

if __name__ == "__main__":
    main()
