I turned the attached context into a clean implementation guide outline, and the source docs confirm the project is a lazy DuckDB-to-Parquet pipeline for AMI water and climate data, with a conditional diffusion model recommended for synthesis. Below is a markdown file you can save as `IMPLEMENTATION_GUIDE.md`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/94422327/21219d61-266d-4e97-b609-71c96dce7b5f/HOW_TO_IMPLEMENT.md?AWSAccessKeyId=ASIA2F3EMEYE3MWWOAO7&Signature=v7JiKcb2PhfpGOCEV5FtJ4LujUg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJGMEQCIF3JtLmINF%2FJA5grnHSbxVDrfjjAnRVXCtDVObhlnlZ1AiANPXQGjtuTVCD1zHQfc9JOh7bRJrL7BhfSAi%2BWF3KGdCrzBAgSEAEaDDY5OTc1MzMwOTcwNSIMP79FCcoxAu81vAKiKtAEdTp1wTwilzApmWxmFjWN8fMPsTrFXPbqq4L2848b9yNjzGWqQ2BxqeFWjLCb42xEph9RAg9PbJsDAC6F4sbmu93HzB55ZO3PuzPsXKWt3iF3E%2F5f6iL9VRdwRjGh8nlh18Xg1Lpoiyr0scOCU18kXw2kuX%2BU3y0j7GMOB40sOcKySjTT4gcRSEJNLHybBD49hH8GwCKZWXKKqoNk7YGuRz5r10hq5k%2F7TwZZlcLyy4qkWNa6f6JwguzsPKG7HrVZOxpTXewJs5XiGE5iNJpQy490D7dS%2BVRBwR%2FwG538QyuZXH%2F%2BkGB24KPGMU%2F09othbkqceNWc5fo1umYaU8cn%2BCMQrowfWgEIQbzFEExNz1vVbF%2BcHP%2F5i3pfcKCTqUNCNrR1kJgtO8eusc7FETMosmSPl6i5PoV54a7RQlBVAx2u6xeQYMRLCZ8lGHOzyzViqxDABoMC8cmnzsIbaFXi8kX53rZcAIueJ7rFtjcv5KwhNgXoiqvHEK0eerL13dY8dwsQP43ZdVsccmgHChjvvVdT4ymBZzjl344PYfvXDhiEdm%2F00irCDNL3wG4oqMgb4Tz4KrDeyBG3zV66AFzNkCEDl64k%2Fr3UJQgL2ExW1Q%2FtbZVqds3Y2bzh8NYnaUe9rE3VdxcnJlKvdW4vXF4BLG0SsrPLOmcRo6ae3O6YAMHOmcaRvRnQpjJNDFWhokxi5yB2LBTIW6KzwrsEb0CKSyQJyQ9yg8%2FM05hB8nQgtiz%2FDi3e%2BVJIDe4XeJbKD6WzfY32DwISZqRqU4HKvt%2B%2BqjC8is7PBjqZASrQ04dDLtlJuphVWpuLMq%2B1oo8GYWQODV2j0E1l3OxmiA7N0GeXQlSt4xKAavrfQjImWbN9O5HCml7uP9wEIPpaa05KY2wPuoUjZcCrdFPuotbe%2Ffd6nUi%2BAyeaheSqZZaDG79UfjP58Fg8EhqaQKr4DlIrseNzL%2Fg44KeHPT9eauu0hqgpW2Tkafqn6BkDEPrOFST4QoDtZw%3D%3D&Expires=1777569803)

```markdown
# Implementation Guide for AMI Synthetic Data Pipeline

This document gives a step-by-step plan for a coding assistant to implement the project described in the repository context. It is written for the existing architecture: lazy DuckDB access, per-meter Parquet storage, weather-enriched panel data, and a conditional diffusion model trained with PyTorch Lightning on Apple Silicon MPS [@README.md][@HOW_TO_IMPLEMENT.md].

## 1. Project Goal

Build an end-to-end pipeline that transforms raw AMI water usage and weather data into a high-resolution training dataset, then trains a conditional generative model that can produce realistic synthetic water demand sequences conditioned on calendar and climate features [@README.md][@HOW_TO_IMPLEMENT.md].

The implementation must preserve bursty usage patterns, avoid loading the full dataset into memory, and support incremental preprocessing across many meters [@HOW_TO_IMPLEMENT.md][@README.md].

## 2. Target Architecture

The system should be implemented as four scripts:

1. `scripts/01_build_features.R`
2. `scripts/02_make_windows.R`
3. `scripts/03_export_to_parquet.py`
4. `scripts/04_train_diffusion.py` [@HOW_TO_IMPLEMENT.md].

The data flow is:

- DuckDB-backed raw and enriched panel data.
- R feature engineering in a lazy, meter-by-meter workflow.
- Window extraction into compact tensor-ready tabular files.
- PyTorch Dataset loading from Parquet.
- Conditional diffusion training using a 1D U-Net backbone with FiLM conditioning [@README.md][@HOW_TO_IMPLEMENT.md].

## 3. Core Constraints

Implement with the following constraints:

- Keep preprocessing memory-safe for a 16 GB machine.
- Process one meter at a time.
- Use parallelization across CPU cores only where it does not cause memory pressure.
- Use MPS for model training on Apple Silicon.
- Avoid any step that requires loading the full population into RAM.
- Persist intermediate outputs to disk rather than accumulating them in memory [@HOW_TO_IMPLEMENT.md][@README.md].

## 4. Data Model

Each row in the enriched panel represents a 15-minute observation with meter usage plus climate variables such as temperature, precipitation, and snowfall [@README.md].

The training setup should build daily windows of length 96, corresponding to 96 fifteen-minute intervals per day [@HOW_TO_IMPLEMENT.md].

The recommended conditioning vector should include:

- Cyclic time features for month, day of week, and hour.
- Weather memory features such as lagged temperature, precipitation accumulation, snowfall flags, and degree-day style transforms.
- Optional indoor/outdoor feature separation if the pipeline supports dual-stream conditioning [@HOW_TO_IMPLEMENT.md].

## 5. Script 01: Feature Engineering in R

### Purpose

Create a meter-wise feature engineering pipeline in R that reads from DuckDB lazily, computes engineered variables, and writes one Parquet file per meter [@HOW_TO_IMPLEMENT.md].

### Required behavior

1. Connect to DuckDB in read-only mode.
2. Query the unified panel view lazily.
3. Enumerate meter IDs first.
4. Loop over meter IDs one at a time.
5. Pull only one meter into memory for feature engineering.
6. Sort by timestamp.
7. Compute lag features, rolling features, and calendar features.
8. Write output as one Parquet file per meter.
9. Keep every transformation incremental and disk-backed [@HOW_TO_IMPLEMENT.md][@README.md].

### Implementation notes

Use a function such as `process_meter(meter_id)` that:

- filters to one meter,
- collects only that meter’s records,
- engineers features,
- validates completeness,
- writes the result to `data/features/meter_<id>.parquet` [@HOW_TO_IMPLEMENT.md].

If the meter set is large, parallelize with `future.apply` or a similar package, but keep worker count conservative to avoid memory spikes [@HOW_TO_IMPLEMENT.md].

### Output columns

At minimum, the engineered dataset should contain:

- `meterid`
- `datetime`
- `usage`
- `tempc`
- `precipmm`
- `snowcm`
- calendar encodings
- lagged usage statistics
- lagged weather features
- missingness flags if needed [@README.md][@HOW_TO_IMPLEMENT.md]

## 6. Script 02: Window Extraction

### Purpose

Convert meter-level feature files into fixed-length daily windows suitable for model training [@HOW_TO_IMPLEMENT.md].

### Required behavior

1. Read each meter Parquet file independently.
2. Construct rolling windows of length 96.
3. Build paired tensors:
   - `x`: target usage sequence.
   - `c`: conditioning vector.
4. Avoid holding all windows from all meters in memory.
5. Append windows incrementally to output storage [@HOW_TO_IMPLEMENT.md].

### Recommended window schema

Each training sample should contain:

- `x` with shape `(96,)` or `(1, 96)` after loading into PyTorch.
- `c` as the conditioning vector.
- Optional split conditioning fields if using indoor/outdoor pathways [@HOW_TO_IMPLEMENT.md].

### Output format

Write compact Parquet files or a dataset directory with one shard per source meter or batch of meters. The output should be immediately readable by PyArrow or Pandas without additional joins [@HOW_TO_IMPLEMENT.md].

## 7. Script 03: Export for PyTorch

### Purpose

Expose the windowed dataset as an efficient PyTorch-compatible source [@HOW_TO_IMPLEMENT.md].

### Required behavior

1. Read Parquet lazily with PyArrow Dataset.
2. Build an iterable dataset.
3. Convert batches to tensors on demand.
4. Yield only what training needs.
5. Avoid full-dataset loading [@HOW_TO_IMPLEMENT.md].

### Suggested dataset contract

Each yielded example should include:

- `x`: float32 tensor, shape `[1, 96]`.
- `c`: float32 tensor, shape `[cond_dim]`.
- Optional additional splits if the model uses separate indoor and outdoor conditioning streams [@HOW_TO_IMPLEMENT.md].

### DataLoader settings

Use a modest number of workers and keep batch sizes small enough for MPS training stability. Use float32 precision and avoid unnecessary pinned-memory complexity on Apple Silicon [@HOW_TO_IMPLEMENT.md].

## 8. Script 04: Train Diffusion Model

### Purpose

Train a conditional 1D diffusion model that generates realistic usage sequences conditioned on the engineered features [@HOW_TO_IMPLEMENT.md].

### Why diffusion

The project context recommends diffusion over VAE-style models because the data contain flat regions, sharp bursts, zero inflation, and strong timing structure. Diffusion is better suited to preserving burst magnitude and spike timing [@HOW_TO_IMPLEMENT.md].

### Model components

Implement these parts:

1. Conditioning encoder.
2. FiLM layer for conditioning injection.
3. Residual 1D convolution blocks.
4. A 1D U-Net backbone.
5. Diffusion noise schedule.
6. Training wrapper with noise-prediction loss [@HOW_TO_IMPLEMENT.md].

### Conditioning encoder

Encode the conditioning vector `c` into a fixed-width embedding with a small MLP and normalization [@HOW_TO_IMPLEMENT.md].

### FiLM conditioning

Use Feature-wise Linear Modulation to inject conditioning into hidden channels via learned scale and shift terms [@HOW_TO_IMPLEMENT.md].

### Residual block

Use 1D convolutions with SiLU activations and conditioning-aware modulation inside each residual block [@HOW_TO_IMPLEMENT.md].

### U-Net backbone

Use a shallow 1D U-Net suited to 96-step sequences. The architecture in the context uses downsampling, upsampling, skip connections, and a final 1x1 projection to one channel [@HOW_TO_IMPLEMENT.md].

### Training objective

Train the model to predict noise added to the clean sequence under the forward diffusion process using mean squared error loss [@HOW_TO_IMPLEMENT.md].

## 9. Training Setup

Use PyTorch Lightning with MPS acceleration on Apple Silicon [@HOW_TO_IMPLEMENT.md].

Recommended settings:

- accelerator: `mps`
- devices: `1`
- precision: `32`
- optimizer: Adam
- learning rate: `1e-4`
- epochs: around `50` to `100`
- batch size: `32` to `64`, depending on memory [@HOW_TO_IMPLEMENT.md].

Keep the model and batch sizes modest enough to avoid MPS instability. Use float32 throughout unless a later benchmark shows a safe alternative [@HOW_TO_IMPLEMENT.md].

## 10. Sampling Procedure

Implement reverse diffusion sampling after training:

1. Start from Gaussian noise shaped like a daily sequence.
2. Iterate backward across diffusion steps.
3. Condition on the desired calendar and weather vector.
4. Save generated sequences for evaluation [@HOW_TO_IMPLEMENT.md].

This should support scenario-based sampling, meaning the same model can generate usage under different weather or temporal conditions by changing the conditioning vector [@HOW_TO_IMPLEMENT.md].

## 11. Evaluation Plan

Evaluate the generated sequences using the following checks:

- Mean and variance matching.
- Autocorrelation similarity.
- Burst magnitude fidelity.
- Event frequency and inter-arrival structure.
- Weather-conditional behavior consistency [@HOW_TO_IMPLEMENT.md].

Also include sanity checks before full training:

1. Overfit a tiny batch.
2. Verify conditioning split behavior if indoor/outdoor features are used.
3. Confirm the output shape and loss decrease on a small sample [@HOW_TO_IMPLEMENT.md].

## 12. Implementation Order

Follow this order:

1. Implement `01_build_features.R`.
2. Validate one meter end-to-end.
3. Implement `02_make_windows.R`.
4. Confirm window shapes and row counts.
5. Implement `03_export_to_parquet.py`.
6. Confirm the PyTorch dataset yields correct tensors.
7. Implement `04_train_diffusion.py`.
8. Overfit a tiny batch.
9. Scale to full training [@HOW_TO_IMPLEMENT.md].

## 13. Practical Safety Rules

Do not violate these rules:

- Never call `collect()` on the full panel.
- Never stack all meter data into one in-memory object.
- Never build all rolling windows before writing output.
- Never train before validating a single-meter path end-to-end.
- Never increase model complexity before the baseline diffusion path works [@README.md][@HOW_TO_IMPLEMENT.md].

## 14. Minimum Acceptance Criteria

The implementation is complete when:

- The pipeline runs meter-by-meter without memory overflow.
- Feature files are written per meter.
- Window files are written in compact training format.
- The PyTorch dataset streams data lazily.
- The diffusion model trains on MPS.
- Sampling produces realistic 96-step daily usage sequences.
- Evaluation shows burst and seasonality patterns are preserved [@HOW_TO_IMPLEMENT.md][@README.md].

## 15. Notes for the Coding Assistant

Implement the simplest correct version first. Prioritize correctness, memory safety, and reproducibility over model sophistication. If a choice is required between a more expressive architecture and a more stable one, choose the stable option first and extend later [@HOW_TO_IMPLEMENT.md].

Keep each script executable on its own, with clear inputs, outputs, and validation checks. Each stage should be independently rerunnable without requiring a full rebuild of earlier stages unless the source data changed [@README.md][@HOW_TO_IMPLEMENT.md].
```

The guide reflects the repository’s lazy DuckDB/Parquet architecture and the recommendation to use a conditional diffusion model with MPS-friendly training. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/94422327/166af09d-a54c-4bd0-b4a5-3ea1d453a501/README.md?AWSAccessKeyId=ASIA2F3EMEYE3MWWOAO7&Signature=wJrYepZna4XAvM1Qmqrx40NHSAQ%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJGMEQCIF3JtLmINF%2FJA5grnHSbxVDrfjjAnRVXCtDVObhlnlZ1AiANPXQGjtuTVCD1zHQfc9JOh7bRJrL7BhfSAi%2BWF3KGdCrzBAgSEAEaDDY5OTc1MzMwOTcwNSIMP79FCcoxAu81vAKiKtAEdTp1wTwilzApmWxmFjWN8fMPsTrFXPbqq4L2848b9yNjzGWqQ2BxqeFWjLCb42xEph9RAg9PbJsDAC6F4sbmu93HzB55ZO3PuzPsXKWt3iF3E%2F5f6iL9VRdwRjGh8nlh18Xg1Lpoiyr0scOCU18kXw2kuX%2BU3y0j7GMOB40sOcKySjTT4gcRSEJNLHybBD49hH8GwCKZWXKKqoNk7YGuRz5r10hq5k%2F7TwZZlcLyy4qkWNa6f6JwguzsPKG7HrVZOxpTXewJs5XiGE5iNJpQy490D7dS%2BVRBwR%2FwG538QyuZXH%2F%2BkGB24KPGMU%2F09othbkqceNWc5fo1umYaU8cn%2BCMQrowfWgEIQbzFEExNz1vVbF%2BcHP%2F5i3pfcKCTqUNCNrR1kJgtO8eusc7FETMosmSPl6i5PoV54a7RQlBVAx2u6xeQYMRLCZ8lGHOzyzViqxDABoMC8cmnzsIbaFXi8kX53rZcAIueJ7rFtjcv5KwhNgXoiqvHEK0eerL13dY8dwsQP43ZdVsccmgHChjvvVdT4ymBZzjl344PYfvXDhiEdm%2F00irCDNL3wG4oqMgb4Tz4KrDeyBG3zV66AFzNkCEDl64k%2Fr3UJQgL2ExW1Q%2FtbZVqds3Y2bzh8NYnaUe9rE3VdxcnJlKvdW4vXF4BLG0SsrPLOmcRo6ae3O6YAMHOmcaRvRnQpjJNDFWhokxi5yB2LBTIW6KzwrsEb0CKSyQJyQ9yg8%2FM05hB8nQgtiz%2FDi3e%2BVJIDe4XeJbKD6WzfY32DwISZqRqU4HKvt%2B%2BqjC8is7PBjqZASrQ04dDLtlJuphVWpuLMq%2B1oo8GYWQODV2j0E1l3OxmiA7N0GeXQlSt4xKAavrfQjImWbN9O5HCml7uP9wEIPpaa05KY2wPuoUjZcCrdFPuotbe%2Ffd6nUi%2BAyeaheSqZZaDG79UfjP58Fg8EhqaQKr4DlIrseNzL%2Fg44KeHPT9eauu0hqgpW2Tkafqn6BkDEPrOFST4QoDtZw%3D%3D&Expires=1777569803)