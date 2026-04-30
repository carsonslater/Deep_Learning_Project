import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.dataset as ds
import numpy as np

class WaterDataset(IterableDataset):
    def __init__(self, data_dir="data/windows/", batch_size=1024):
        super().__init__()
        # Use pyarrow dataset for lazy loading
        self.dataset = ds.dataset(data_dir, format="parquet")
        self.batch_size = batch_size

    def __iter__(self):
        # Iterate over the dataset in batches
        for batch in self.dataset.to_batches(batch_size=self.batch_size):
            df = batch.to_pandas()

            # x is a list of arrays in pandas, we need to stack them
            x_list = df["x"].tolist()
            c_in_list = df["c_in"].tolist()
            c_out_list = df["c_out"].tolist()

            # Convert to tensors
            x = torch.tensor(np.stack(x_list), dtype=torch.float32)
            c_in = torch.tensor(np.stack(c_in_list), dtype=torch.float32)
            c_out = torch.tensor(np.stack(c_out_list), dtype=torch.float32)

            # PyTorch expects shape [Batch, Channels, Sequence_Length]
            # So we add a channel dimension for x: [B, 1, 96]
            x = x.unsqueeze(1)

            # Yield single samples instead of batches if DataLoader is used with batch_size, 
            # but since we already batched using PyArrow, we can yield batches directly if DataLoader batch_size=None
            for i in range(x.shape[0]):
                yield x[i], c_in[i], c_out[i]

def get_dataloader(data_dir="data/windows/", batch_size=64, num_workers=4):
    """
    Returns a DataLoader for the water usage dataset.
    """
    # Create the iterable dataset. We use a larger batch size for reading from PyArrow
    # to be efficient, but the DataLoader will collate them into the requested batch_size.
    dataset = WaterDataset(data_dir=data_dir, batch_size=2048)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,   
        num_workers=num_workers,     
        pin_memory=False         # MPS doesn't benefit from pinned memory
    )
    return loader

if __name__ == "__main__":
    # Test the dataloader
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
        print(f"Could not test dataloader (maybe data is missing): {e}")
