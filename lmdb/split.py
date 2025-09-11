"""
The script has the following dependencies:

    pip install lmdb==1.7.3 tqdm

Minimum supported Python version: 3.6

You can also run this script in the
DeepMalNet conda environment.




This script splits all keys in a LMDB
database into N parts, where the i^th
part contains K_i keys. That means

    N
    sum(K_i) = K
    i=1

where K is the total number of keys
in the database.

This could be used to split a LMDB
dataset into train, cv and test sets.

Usage:

    python lmdb/split.py /path/to/lmdb /path/to/output/dir p_1 p_2 p_3...

where

    p_i = K_i / K

For example, to split a dataset into
70% train, 10% cv, 20% test:

    python lmdb/split.py /path/to/your/lmdb /path/to/your/output/dir 0.7 0.1 0.2

The results would be 3 files:

    p1.txt
    p2.txt
    p3.txt

You could then rename them to
train_keys.txt, cv_keys.txt,
test_keys.txt for example.

Each file contains the keys in the
following format:

    b'key1',
    b'key2',
    b'key3',
    ...

That is, the keys are encoded into
Python `bytes` objects using the
`repr` function; they are separated
by comma + newline combo(s).

To load that back into a Python set:

with open("p0.txt", 'r') as f:
    p0 = eval("{" + f.read() + "}")

Beware, for a file that contains 320,000
keys, each key is 32 bytes long, then
the total memory occupied by that set
alone is about 400 MB. It is recommended
to run the garbage collector after each
time such a file is loaded:

    import gc
    gc.collect()

You can read the file another way,
e.g. random access, line-by-line...
if you feel the need to.
"""

import os, sys
import lmdb
from tqdm import tqdm

def run(lmdb_path, output_dir_path, args):
    # type: (str, str, list[str]) -> None
    assert len(args) > 0, "No proportions?"

    proportions = list(map(float, args))
    S_p = sum(proportions)
    assert S_p >= 0.99 and S_p <= 1.01, "Sum of proportions is not (or not close to) 1.00"
    assert all(map(lambda x: x > 0, proportions)), "Some proportions are not positive???"

    os.makedirs(output_dir_path, exist_ok=True)

    env = lmdb.open(
        lmdb_path,
        map_size=1024 * 1024 * 1024 * 1024, # 1 TB
        readonly=True,
        lock=False,
    )

    counts = [] # type: list[int]
    K = int(env.stat()["entries"])
    for i in range(len(proportions)):
        p = proportions[i]
        is_last_proportion = (i == len(proportions) - 1)
        if is_last_proportion:
            k_i = K - sum(counts)
        else:
            k_i = int(K * p)
        counts.append(k_i)
    
    assert sum(counts) == K
    print("Number of entries per parts:", "|".join(map(str, counts)))

    files = []
    try:
        for i in range(len(proportions)):
            files.append(
                open(f"{output_dir_path}/p{i}.txt", 'wb')
            )

        with tqdm(total=K) as pbar:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                i = 0
                num_entries_left = counts[0]
                for key, _value in cursor:
                    assert isinstance(key, bytearray) or isinstance(key, bytes)

                    num_entries_left -= 1
                    if num_entries_left < 0:
                        files.pop(0).close()
                        i += 1
                        if i >= len(counts):
                            break
                        num_entries_left = counts[i] - 1

                    f = files[0]
                    f.write(repr(key).encode('ascii') + b',\n')
                    pbar.update(1)
    finally:
        map(lambda f: f.close(), files)

def main():
    if len(sys.argv) < 4:
        raise RuntimeError("Insufficient arguments")
    
    _argv0, lmdb_path, output_dir_path, *proportions = sys.argv
    run(lmdb_path, output_dir_path, proportions)

if __name__ == "__main__":
    main()
