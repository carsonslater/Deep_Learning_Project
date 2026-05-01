# Diffusion Model Fix: Implementation Guide

## The Definitive Answer

Both the "quick fixes" (scaling + rolling windows) and the dual-head architecture are
required. They are not competing solutions — they fix different layers of the same problem:

| Layer | Problem | Solution |
|---|---|---|
| Mathematics | Gaussian noise cannot corrupt unscaled signal (magnitude ~900) | Log-scaling + z-normalisation (Fix 1) |
| Data | Disjoint windows starve the model of spike examples | Overlapping stride windows (Fix 2) |
| Architecture | Single continuous head cannot model zero-inflated distribution | Dual-head U-Net (Fix 3) |

**Implement in this exact order.** Fix 1 and Fix 2 are prerequisites. Without them, Fix 3
will collapse identically to the current model. After each phase, do a short validation run
(5–10 epochs) before proceeding to the next.

---

## Phase 1 — Data Pipeline Fixes (Prerequisite)

These changes happen in the dataset/dataloader code before any model is touched.

### Step 1.1 — Compute and persist normalisation statistics

Do this once over the training split. Never compute stats over the full dataset (data leakage).

```python
import numpy as np
import json
from pathlib import Path

def compute_normalisation_stats(parquet_dir: str, output_path: str = "norm_stats.json"):
    """
    Compute log1p mean/std of NON-ZERO usage values only.
    Zeros are structural (no flow) and must not influence the scale.
    """
    import duckdb
    con = duckdb.connect()

    # Pull only non-zero usage values — never load full dataset into RAM
    result = con.execute(f"""
        SELECT
            AVG(log_usage)  AS log_mean,
            STDDEV(log_usage) AS log_std
        FROM (
            SELECT LN(usage_gallons + 1.0) AS log_usage
            FROM read_parquet('{parquet_dir}/train/**/*.parquet')
            WHERE usage_gallons > 0.0
        )
    """).fetchone()

    stats = {"log_mean": result[0], "log_std": result[1]}
    Path(output_path).write_text(json.dumps(stats, indent=2))
    print(f"Stats: log_mean={stats['log_mean']:.4f}, log_std={stats['log_std']:.4f}")
    return stats
```

**Expected output:** `log_mean` will be approximately 1.5–3.5 depending on your population.
`log_std` will be approximately 0.8–1.8. If either is outside these ranges, check for
unit errors in your raw data.

---

### Step 1.2 — Rewrite the transform function

Replace whatever scaling currently exists in your dataset with this:

```python
import numpy as np

def transform_usage_sequence(raw_sequence: np.ndarray, log_mean: float, log_std: float):
    """
    Transform a raw 96-step usage sequence into two tracks:
      - occurrence_mask : float32 array of shape (96,), values in {0.0, 1.0}
      - log_magnitude   : float32 array of shape (96,), z-scored log values,
                          0.0 where occurrence_mask is 0.0

    This decomposition is essential. A single continuous transform cannot
    represent a zero-inflated distribution — the zeros are structural, not
    'small values', and must be separated before diffusion.
    """
    eps = 1e-6

    # Binary occurrence mask — true wherever flow occurred
    occurrence_mask = (raw_sequence > 0.0).astype(np.float32)

    # Log-transform and z-score non-zero values only
    log_vals = np.log1p(raw_sequence)  # log(x + 1), safe for x=0
    log_normalised = np.where(
        occurrence_mask > 0,
        (log_vals - log_mean) / (log_std + eps),
        0.0
    ).astype(np.float32)

    return occurrence_mask, log_normalised


def inverse_transform(occurrence_mask: np.ndarray,
                      log_magnitude: np.ndarray,
                      log_mean: float,
                      log_std: float) -> np.ndarray:
    """
    Reconstruct gallons from model outputs at inference time.
    Apply a Bernoulli gate from the occurrence head first