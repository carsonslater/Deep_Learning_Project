import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.dataset as ds
import numpy as np
import math

class WaterDataset(IterableDataset):
    def __init__(self, data_dir="data/windows/", batch_size=512):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = ds.dataset(data_dir, format="parquet")
        self.batch_size = batch_size
        
        # Load normalization stats dynamically
        import json
        with open("norm_stats.json", "r") as f:
            stats = json.load(f)
            for key, val in stats.items():
                setattr(self, key, val)

    def __len__(self):
        # Extremely fast metadata read to get total rows
        total_rows = sum(fragment.metadata.num_rows for fragment in self.dataset.get_fragments())
        return total_rows

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
            # Optimized Arrow-to-NumPy conversion
            # Use to_numpy(zero_copy_only=False) to ensure we handle any internal Arrow structures safely
            raw_x = batch.column("x").flatten().to_numpy().astype(np.float32).reshape(-1, 96)
            c_in_arr = batch.column("c_in").flatten().to_numpy().astype(np.float32).reshape(-1, 12, 96)
            c_out_arr = batch.column("c_out").flatten().to_numpy().astype(np.float32).reshape(-1, 10, 96)
            
            # Apply Normalization to target x
            log_mean = self.log_mean
            log_std = self.log_std
            eps = 1e-6
            
            occurrence_mask = (raw_x > 0.0).astype(np.float32)
            log_vals = np.log1p(np.maximum(raw_x, 0.0)) # Ensure non-negative for log
            log_normalised = np.where(
                occurrence_mask > 0,
                (log_vals - log_mean) / (log_std + eps),
                0.0
            ).astype(np.float32)
            
            x_arr = np.stack([occurrence_mask, log_normalised], axis=1)
            
            # Apply Normalization to c_in (Lagged usage features: indices 6-11)
            # Masking to prevent NaN/Inf from log1p(0)
            lag_usage = c_in_arr[:, 6:12, :]
            lag_mask = (lag_usage > 0.0).astype(np.float32)
            c_in_arr[:, 6:12, :] = np.where(
                lag_mask > 0,
                (np.log1p(np.maximum(lag_usage, 0.0)) - log_mean) / (log_std + eps),
                0.0
            )
            
            # Apply Normalization to c_out (Weather features)
            weather_cols = ["temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h", "precip_3d", "snow_24h", "gdd_7d"]
            weather_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
            
            for i, col in enumerate(weather_cols):
                mean = getattr(self, f"{col}_mean")
                std = getattr(self, f"{col}_std")
                idx = weather_indices[i]
                c_out_arr[:, idx, :] = (c_out_arr[:, idx, :] - mean) / (std + eps)

            # Convert to torch
            x = torch.from_numpy(x_arr)
            c_in = torch.from_numpy(c_in_arr)
            c_out = torch.from_numpy(c_out_arr)
            
            # Final Integrity Check: Drop any sample with NaNs or Infs
            for i in range(x.shape[0]):
                if torch.isnan(x[i]).any() or torch.isinf(x[i]).any(): continue
                if torch.isnan(c_in[i]).any() or torch.isinf(c_in[i]).any(): continue
                if torch.isnan(c_out[i]).any() or torch.isinf(c_out[i]).any(): continue
                
                yield x[i].clone(), c_in[i].clone(), c_out[i].clone()

def get_dataloader(data_dir="data/windows/", batch_size=32, num_workers=0):
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

