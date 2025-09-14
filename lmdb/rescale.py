import torch
import numpy as np
import lmdb, msgpack, msgpack_numpy
from tqdm import tqdm
import os
import multiprocessing as mp
import traceback

msgpack_numpy.patch()

class Batching:
    def __init__(self, on_flush, batch_size=64):
        self.on_flush = on_flush
        self.batch = []
        self.batch_size = batch_size
    
    def add(self, item):
        self.batch.append(item)
        if len(self.batch) >= self.batch_size:
            self._flush()
    
    def add_all(self, items):
        self.batch.extend(items)
        if len(self.batch) >= self.batch_size:
            self._flush()
    
    def finalize(self):
        self._flush()

    def _flush(self):
        if len(self.batch) == 0:
            return
        i = 0
        while i < len(self.batch):
            j = min(len(self.batch), i + self.batch_size)
            self.on_flush(self.batch[i:j])
            i = j
        self.batch.clear()

def load_scaling(scaling_params_file_path):
    data = np.load(scaling_params_file_path)
    mean = torch.from_numpy(data["mean"]).float()
    std = torch.from_numpy(data["std"]).float()

    return mean, std

class ComputeMeanStdStableFatal(Exception):
    pass

class EmptyDB(ComputeMeanStdStableFatal):
    pass

class WrongStartOffset(ComputeMeanStdStableFatal):
    pass

class ProgressPoint:
    # sum all this must be 10
    UNPACK = 2
    RESCALE = 4
    SAVE = 4

def rescale(input_lmdb_path, output_lmdb_path, scaling_file, start, stride, total, progress_queue):
    """
    Apply the rescaling. Done on CPU.
    """
    mean, std = load_scaling(scaling_file)

    input_env = lmdb.open(
        input_lmdb_path,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=65536,
    )

    output_env = lmdb.open(
        output_lmdb_path,
        map_size=1024 * 1024 * 1024 * 1024, # 1 TB
        subdir=True,
        readonly=False,
        lock=True,
        writemap=False,
    )

    unpackb = msgpack.unpackb

    def write_batch(batch):
        with output_env.begin(write=True) as write_txn:
            for k, v in batch:
                write_txn.put(k, v)
                progress_queue.put(ProgressPoint.SAVE)
    write_batching = Batching(write_batch)
    
    def rescale_batch(batch):
        x_list = [torch.tensor(item[1]['ef']) for item in batch]
        x_stacked = torch.stack(x_list)
        x_rescaled = (x_stacked - mean.unsqueeze(0)) / std.unsqueeze(0)

        if x_rescaled.device.type == 'cpu' and not x_rescaled.requires_grad:
            x_rescaled = x_rescaled.numpy()
        else:
            x_rescaled = x_rescaled.detach().cpu().numpy()

        write_batching.add_all((

            batch[i][0],

            msgpack.packb({
                'lb': batch[i][1]['lb'],
                'ef': x_rescaled[i],
            }, use_bin_type=True),

        ) for i in range(len(batch)))
        progress_queue.put(ProgressPoint.RESCALE * len(batch))
    rescale_batching = Batching(rescale_batch)

    try:
        with input_env.begin(write=False) as readonly_txn:
            cursor = readonly_txn.cursor()
            if not cursor.first():
                raise EmptyDB()
            for _ in range(start):
                if not cursor.next():
                    raise WrongStartOffset(start)
            
            idx = start
            while idx < total:
                key = cursor.key()
                raw_value = cursor.value()
                old_payload = unpackb(raw_value, raw=False)
                progress_queue.put(ProgressPoint.UNPACK)
                rescale_batching.add((key, old_payload))

                for _ in range(stride):
                    if not cursor.next():
                        break
                idx += stride

        rescale_batching.finalize()
        write_batching.finalize()
    except ComputeMeanStdStableFatal:
        pass
    except Exception:
        print(f"Exception in worker with offset {start}: {traceback.format_exc()}")
    finally:
        input_env.close()
        output_env.close()

def rescale_parallel(input_lmdb_path, output_lmdb_path, scaling_file, num_workers=-1):
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)
    
    input_env = lmdb.open(
        input_lmdb_path,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=65536,
    )
    total = int(input_env.stat()["entries"])
    input_env.close()

    output_env = lmdb.open(
        output_lmdb_path,
        map_size=1024 * 1024 * 1024 * 1024, # 1 TB
        subdir=True,
        readonly=False,
        lock=True,
        writemap=False,
    )
    output_env.close()

    print(f"Rescaling LMDB: {input_lmdb_path}")
    print(f"to target LMDB: {output_lmdb_path}")
    print(f"    with num_workers = {num_workers}")
    print(f"------------------------------------")
    print()

    if total == 0:
        raise ValueError("LMDB is empty")

    manager = mp.Manager()
    progress_queue = manager.Queue()

    with mp.Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(rescale, (input_lmdb_path, output_lmdb_path, scaling_file, start, num_workers, total, progress_queue))
            for start in range(num_workers)
        ]

        # tqdm progress in main process
        total_points = total * 10
        with tqdm(total=total_points) as pbar:
            done_points = 0
            while done_points < total_points:
                advance_points = progress_queue.get()
                done_points += advance_points
                pbar.update(advance_points)

        results = [r.get() for r in results]

def main():
    import sys
    if len(sys.argv) != 4:
        raise RuntimeError(f"Exactly 4 arguments are expected, got {len(sys.argv)} instead")

    print("sys.argv =", sys.argv)
    _argv0, input_lmdb_path, output_lmdb_path, scaling_params_dir_path = sys.argv
    scaling_params_file_path = os.path.join(scaling_params_dir_path, "scaling.npz")
    rescale_parallel(input_lmdb_path, output_lmdb_path, scaling_params_file_path)

if __name__ == "__main__":
    main()
