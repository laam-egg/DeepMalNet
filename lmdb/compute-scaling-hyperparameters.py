import torch
import numpy as np
import lmdb, msgpack, msgpack_numpy
from tqdm import tqdm
import os
import multiprocessing as mp

msgpack_numpy.patch()

OUTPUT_DTYPE = torch.float64

class ComputeMeanStdStableFatal(Exception):
    pass

class EmptyDB(ComputeMeanStdStableFatal):
    pass

class WrongStartOffset(ComputeMeanStdStableFatal):
    pass

def compute_mean_std_stable(lmdb_path, start, stride, total, progress_queue):
    """
    Using Welford's online algorithm for mean and variance,
    preventing arithmetic overflow
    """
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=65536,
    )
    unpackb = msgpack.unpackb
    n = 0
    mean = None
    M2 = None  # sum of squares of differences from the mean

    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            if not cursor.first():
                raise EmptyDB()
            for _ in range(start):
                if not cursor.next():
                    raise WrongStartOffset(start)
            
            idx = start
            while idx < total:
                raw_value = cursor.value()
                payload = unpackb(raw_value, raw=False)
                x = torch.tensor(payload['ef'], dtype=torch.float64)

                if mean is None:
                    mean = torch.zeros_like(x)
                    M2 = torch.zeros_like(x)

                n += 1
                delta = x - mean
                mean = mean + delta / n
                delta2 = x - mean
                M2 = M2 + delta * delta2

                progress_queue.put(1)

                for _ in range(stride):
                    if not cursor.next():
                        break
                idx += stride

        return mean, M2, n
    except ComputeMeanStdStableFatal as e:
        return None, None, 0
    finally:
        env.close()

def compute_mean_std_parallel(lmdb_path, num_workers=-1, output_dtype=torch.float32):
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=65536,
    )
    total = int(env.stat()["entries"])
    env.close()

    print(f"Analyzing LMDB: {lmdb_path}")
    print(f"    with num_workers = {num_workers}")
    print(f"------------------------------------")
    print()

    if total == 0:
        raise ValueError("LMDB is empty")

    manager = mp.Manager()
    progress_queue = manager.Queue()

    with mp.Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(compute_mean_std_stable, (lmdb_path, start, num_workers, total, progress_queue))
            for start in range(num_workers)
        ]

        # tqdm progress in main process
        with tqdm(total=total) as pbar:
            done = 0
            while done < total:
                progress_queue.get()
                done += 1
                pbar.update(1)

        results = [r.get() for r in results]

    # Merge results (Welford's merge)
    mean, M2, n = None, None, 0
    for m, M2_local, n_local in results:
        if m is None:
            continue
        if mean is None:
            mean, M2, n = m, M2_local, n_local
        else:
            delta = m - mean
            tot = n + n_local
            mean = mean + delta * (n_local / tot)
            M2 = M2 + M2_local + delta**2 * n * n_local / tot
            n = tot

    if n <= 1:
        raise ValueError(f"n <= 1: n = {n}")
    
    variance = M2 / (n - 1)
    std = torch.sqrt(variance + 1e-8)

    return mean.to(output_dtype), std.to(output_dtype)

def save_scaling_params(mean, std, output_dir_path):
    output_file_path = os.path.join(output_dir_path, "scaling.npz")
    print(f"[INFO] Saving to file: {output_file_path}")
    np.savez(
        output_file_path,
        mean = mean.cpu().numpy(),
        std = std.cpu().numpy(),
    )
    print(f"[INFO] Done saving to file: {output_file_path}")

def main():
    import sys
    if len(sys.argv) != 3:
        raise RuntimeError(f"Exactly 3 arguments are expected, got {len(sys.argv)} instead")

    print("sys.argv =", sys.argv)
    _argv0, lmdb_path, output_dir_path = sys.argv
    mean, std = compute_mean_std_parallel(lmdb_path)
    print(f"mean = {mean}")
    print(f"std = {std}")
    save_scaling_params(mean, std, output_dir_path)

if __name__ == "__main__":
    main()
