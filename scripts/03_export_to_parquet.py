import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.dataset as ds
import numpy as np
import math

class WaterDataset(IterableDataset):
    def __init__(self, data_dir="data/windows/", batch_size=2048):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = ds.dataset(data_dir, format="parquet")
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        all_files = self.dataset.files
        
        if worker_info is None:
            files = all_files
        else:
            # Shard files across workers
            per_worker = int(math.ceil(len(all_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(all_files))
            files = all_files[iter_start:iter_end]
            
        if not files:
            return

        worker_ds = ds.dataset(files, format="parquet")
        
        for batch in worker_ds.to_batches(batch_size=self.batch_size):
            # Use to_pylist() for list columns to avoid ArrowInvalid zero-copy errors
            x_arr = np.array(batch.column("x").to_pylist(), dtype=np.float32)
            c_in_arr = np.array(batch.column("c_in").to_pylist(), dtype=np.float32)
            c_out_arr = np.array(batch.column("c_out").to_pylist(), dtype=np.float32)
            
            # Convert to torch
            x = torch.from_numpy(x_arr)
            if x.ndim == 2: x = x.unsqueeze(1) # Ensure [B, 1, 96]
            
            c_in = torch.from_numpy(c_in_arr).view(-1, 9, 96)
            c_out = torch.from_numpy(c_out_arr).view(-1, 10, 96)
            
            # Yield individual samples (Dataloader will re-batch them)
            for i in range(x.shape[0]):
                yield x[i], c_in[i], c_out[i]

def get_dataloader(data_dir="data/windows/", batch_size=64, num_workers=4):
    dataset = WaterDataset(data_dir=data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,   
        num_workers=num_workers,     
        pin_memory=False
    )
    return loader

if __name__ == "__main__":
    print("Testing DataLoader...")
    try:
        loader = get_dataloader(batch_size=32, num_workers=0)
        for x, c_in, c_out in loader:
            print(f"Batch x shape: {x.shape}")
            print(f"Batch c_in shape: {c_in.shape}")
            print(f"Batch c_out shape: {c_out.shape}")
            break
        print("DataLoader works!")
    except Exception as e:
        print(f"Could not test dataloader: {e}")

