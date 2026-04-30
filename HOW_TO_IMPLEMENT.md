Here is your **revised, implementation-ready Markdown**, keeping the original structure but now **fully integrating the conditional diffusion model**, including mathematical formulation, architecture, and a concrete training setup using **PyTorch Lightning with MPS acceleration**.

---

# AMI Synthetic Data Generation via Conditional Deep Generative Models

## 1. Problem Formulation

We observe a panel dataset of residential water usage:

* Meters: ( i = 1, \dots, N )
* Time (15-min intervals): ( t = 1, \dots, T_i )

Each observation consists of:

[
y_{i,t} \in \mathbb{R}_{\ge 0} \quad \text{(water usage)}
]

[
w_{i,t} = (\text{temp}*{i,t}, \text{precip}*{i,t}, \text{snow}_{i,t}) \in \mathbb{R}^3
]

We construct **daily sequences** of length ( L = 96 ):

[
\mathbf{x}_{i,d} \in \mathbb{R}^{96}
]

paired with a conditioning vector:

[
\mathbf{c}_{i,d} \in \mathbb{R}^{d_c}
]

---

## 2. Conditioning Vector Construction

We define a physically grounded feature map:

[
\mathbf{c}*{i,t} = \phi(w*{i,\cdot}, \text{calendar}_t)
]

### 2.1 Temporal Features (Cyclic Encoding)

[
\sin\left(\frac{2\pi \cdot \text{month}_t}{12}\right), \quad
\cos\left(\frac{2\pi \cdot \text{month}_t}{12}\right)
]

[
\sin\left(\frac{2\pi \cdot \text{dow}_t}{7}\right), \quad
\cos\left(\frac{2\pi \cdot \text{dow}_t}{7}\right)
]

[
\sin\left(\frac{2\pi \cdot \text{hour}_t}{24}\right), \quad
\cos\left(\frac{2\pi \cdot \text{hour}_t}{24}\right)
]

---

### 2.2 Weather Memory Features

[
\text{temp}*{t-1h}, \quad \text{temp}*{t-24h}, \quad \text{temp}_{t-48h}
]

[
\text{precip}*{3d}(t) = \sum*{k=0}^{287} \text{precip}_{t-k}
]

[
\text{snow_flag}_t = \mathbf{1}[\text{snow}_t > 0]
]

[
\text{snow}*{24h}(t) = \sum*{k=0}^{95} \text{snow}_{t-k}
]

[
\text{GDD}*{7d}(t) = \sum*{k=0}^{671} \max(\text{temp}_{t-k} - 10, 0)
]

---

### 2.3 Final Conditioning Vector

[
\mathbf{c}_{i,t} \in \mathbb{R}^{d_c}
]

Concatenation of all features above.

---

## 3. Data Transformation Pipeline (R)

### 3.1 Lazy Data Access (DuckDB)

```r
con <- dbConnect(duckdb(), "data/ami_db.duckdb", read_only = TRUE)
panel <- tbl(con, "climate_data")
```

---

### 3.2 Feature Engineering

(unchanged from previous draft)

---

### 3.3 Window Extraction

We construct:

* ( \mathbf{x} \in \mathbb{R}^{96} )
* ( \mathbf{c} \in \mathbb{R}^{d_c} )

---

### 3.4 Normalization

[
\tilde{x} = \frac{x - \mu}{\sigma}
]

Persist scalers for inference.

---

# 🔄 4. Model: Conditional Diffusion Model

We model:

[
p_\theta(\mathbf{x} \mid \mathbf{c})
]

using a **denoising diffusion probabilistic model (DDPM)**.

---

## 4.1 Forward Diffusion Process

We define a Markov chain:

[
q(\mathbf{x}*t \mid \mathbf{x}*{t-1}) = \mathcal{N}(\mathbf{x}*t; \sqrt{1-\beta_t}\mathbf{x}*{t-1}, \beta_t I)
]

Closed form:

[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}
]

where:

* ( \boldsymbol{\epsilon} \sim \mathcal{N}(0, I) )
* ( \bar{\alpha}*t = \prod*{s=1}^t (1 - \beta_s) )

---

## 4.2 Reverse Process (Learned)

We train a neural network:

[
\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})
]

to predict noise.

---

## 4.3 Training Objective

[
\mathcal{L} =
\mathbb{E}*{t, \mathbf{x}, \boldsymbol{\epsilon}} \left[
| \boldsymbol{\epsilon} - \epsilon*\theta(\mathbf{x}_t, t, \mathbf{c}) |^2
\right]
]

---

# 🧠 5. Architecture: Conditional 1D U-Net

## 5.1 Conditioning Encoder

```python
class ConditioningEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, c):
        return self.net(c)
```

---

## 5.2 FiLM Conditioning

[
h' = \gamma(\mathbf{c}) \odot h + \beta(\mathbf{c})
]

```python
class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, c):
        return self.gamma(c).unsqueeze(-1) * h + self.beta(c).unsqueeze(-1)
```

---

## 5.3 Residual Block

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.film  = FiLM(cond_dim, out_ch)
        self.act   = nn.SiLU()

    def forward(self, x, c):
        h = self.act(self.conv1(x))
        h = self.film(h, c)
        h = self.act(self.conv2(h))
        return h + x
```

---

## 5.4 U-Net Backbone

```python
class UNet1D(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()

        self.down1 = ResBlock(1, 32, cond_dim)
        self.down2 = ResBlock(32, 64, cond_dim)
        self.pool = nn.AvgPool1d(2)

        self.up1 = ResBlock(64, 32, cond_dim)
        self.up2 = ResBlock(32, 32, cond_dim)

        self.out = nn.Conv1d(32, 1, 1)

    def forward(self, x, c):
        d1 = self.down1(x, c)
        d2 = self.down2(self.pool(d1), c)

        u1 = nn.functional.interpolate(d2, scale_factor=2)
        u1 = self.up1(u1, c)

        u2 = self.up2(u1 + d1, c)

        return self.out(u2)
```

---

## 5.5 Full Model

```python
class DiffusionModel(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.encoder = ConditioningEncoder(cond_dim)
        self.unet = UNet1D(cond_dim=64)

    def forward(self, x_t, t, c):
        c_embed = self.encoder(c)
        return self.unet(x_t, c_embed)
```

---

# ⚡ 6. Training Implementation (PyTorch Lightning + MPS)

## 6.1 Lightning Module

```python
import pytorch_lightning as pl

class LitDiffusion(pl.LightningModule):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T

    def training_step(self, batch, batch_idx):
        x, c = batch

        noise = torch.randn_like(x)
        t = torch.randint(0, self.T, (x.size(0),), device=self.device)

        x_t = q_sample(x, t, noise)

        noise_pred = self.model(x_t, t, c)

        loss = ((noise - noise_pred) ** 2).mean()
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
```

---

## 6.2 Trainer (MPS Acceleration)

```python
trainer = pl.Trainer(
    accelerator="mps",   # Apple Silicon GPU
    devices=1,
    max_epochs=50,
    precision=32
)
```

---

## 6.3 Training

```python
model = DiffusionModel(cond_dim=d_c)
lit_model = LitDiffusion(model)

trainer.fit(lit_model, train_dataloader)
```

---

# 🎲 7. Scenario-Based Sampling

```python
x_t = torch.randn(batch_size, 1, 96, device="mps")

for t in reversed(range(T)):
    noise_pred = model(x_t, t, c_scenario)
    x_t = p_sample(x_t, noise_pred, t)
```

---

# 📊 8. Evaluation

### 8.1 Distributional Fidelity

* Mean / variance
* Autocorrelation

### 8.2 Weather-Conditional Fidelity

[
\mathbb{E}[y \mid \text{temp}, \text{precip}]
]

### 8.3 Burst Structure

* Peak magnitude
* Event frequency
* Inter-arrival times


Got it — I’ll **extend your existing markdown spec** with a new section that cleanly drops in the architectural decision layer and gives you implementation-ready detail.

---

# 🔧 9. Model Architecture Selection (Critical Design Layer)

The choice of generative architecture is **not interchangeable** for this problem.
Residential water usage exhibits the following structural properties:

* **Zero-inflation / intermittency** (long flat regions)
* **Impulse-like bursts** (fixtures, irrigation)
* **Strong diurnal periodicity**
* **Weather-driven regime shifts**

Formally, each sequence can be decomposed as:

[
\mathbf{x} = \mathbf{x}^{(\text{indoor})} + \mathbf{x}^{(\text{bursty})} + \mathbf{x}^{(\text{weather})}
]

A single smooth latent representation (e.g., vanilla VAE) tends to **blur this decomposition**, leading to unrealistic synthetic data.

---

## 9.1 Architecture Candidates

### A. Conv1D Conditional VAE (Baseline)

**Structure**

* Encoder: Conv1D → Dense → latent ( (\mu, \sigma) )
* Decoder: Dense → ConvTranspose1D
* Conditioning: FiLM injection

**Strengths**

* Fast
* Stable
* Easy to implement

**Weaknesses**

* Smooths spikes
* Underestimates extremes
* Poor burst timing fidelity

👉 Use as a **baseline only**

---

### B. Transformer-based cVAE

**Structure**

* Encoder/decoder built with self-attention blocks
* Conditioning via cross-attention or FiLM

**Strengths**

* Captures long-range dependencies
* Flexible conditioning

**Weaknesses**

* Sequence length (96) is too short to justify attention overhead
* Requires more data to outperform CNNs
* Slower training

👉 Not recommended for first implementation

---

### C. Conditional Diffusion Model (Recommended)

We instead model:

[
p(\mathbf{x} \mid \mathbf{c}) = \int p(\mathbf{x}*0 \mid \mathbf{x}*T, \mathbf{c}) \prod*{t=1}^{T} p(\mathbf{x}*{t-1} \mid \mathbf{x}*t, \mathbf{c}) d\mathbf{x}*{1:T}
]

where noise is progressively removed.

---

## 9.2 Why Diffusion Wins for This Problem

Diffusion models excel at:

* Preserving **sharp discontinuities**
* Modeling **multi-modal distributions**
* Generating **realistic spike timing**
* Avoiding posterior collapse (common in VAEs)

In your context:

| Feature              | VAE            | Diffusion |
| -------------------- | -------------- | --------- |
| Burst magnitude      | Underestimated | Accurate  |
| Burst timing         | Blurry         | Sharp     |
| Zero regions         | Noisy          | Clean     |
| Weather conditioning | Moderate       | Strong    |

---

## 9.3 Final Recommended Architecture

### Conditional 1D Diffusion with U-Net Backbone

---

### Forward Process

[
\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_0 + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
]

---

### Reverse Model

[
\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})
]

---

### Training Objective

[
\mathcal{L} =
\mathbb{E}*{t, \mathbf{x}, \boldsymbol{\epsilon}} \left[
| \boldsymbol{\epsilon} - \epsilon*\theta(\mathbf{x}_t, t, \mathbf{c}) |^2
\right]
]

---

## 9.4 PyTorch Architecture (Implementation-Ready)

### Conditioning Injection (FiLM)

```python
class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, hidden_dim)
        self.to_beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, c):
        gamma = self.to_gamma(c).unsqueeze(-1)
        beta  = self.to_beta(c).unsqueeze(-1)
        return gamma * h + beta
```

---

### Residual Block

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.film  = FiLM(cond_dim, out_ch)
        self.act   = nn.SiLU()

    def forward(self, x, c):
        h = self.act(self.conv1(x))
        h = self.film(h, c)
        h = self.act(self.conv2(h))
        return h + x
```

---

### U-Net Backbone

```python
class UNet1D(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()

        self.down1 = ResBlock(1, 32, cond_dim)
        self.down2 = ResBlock(32, 64, cond_dim)

        self.pool = nn.AvgPool1d(2)

        self.up1 = ResBlock(64, 32, cond_dim)
        self.up2 = ResBlock(32, 32, cond_dim)

        self.out = nn.Conv1d(32, 1, 1)

    def forward(self, x, c):
        d1 = self.down1(x, c)
        d2 = self.down2(self.pool(d1), c)

        u1 = nn.functional.interpolate(d2, scale_factor=2)
        u1 = self.up1(u1, c)

        u2 = self.up2(u1 + d1, c)

        return self.out(u2)
```

---

### Full Diffusion Model

```python
class ConditionalDiffusion(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.cond_encoder = ConditioningEncoder(cond_dim, 64)
        self.unet = UNet1D(cond_dim=64)

    def forward(self, x_t, t, c):
        c_embed = self.cond_encoder(c)
        return self.unet(x_t, c_embed)
```

---

## 9.5 Training Loop (Simplified)

```python
for x, c in dataloader:

    noise = torch.randn_like(x)
    t = torch.randint(0, T, (x.size(0),))

    x_t = q_sample(x, t, noise)

    noise_pred = model(x_t, t, c)

    loss = ((noise - noise_pred) ** 2).mean()

    loss.backward()
    optimizer.step()
```

---

## 9.6 Scenario-Based Sampling

```python
x_t = torch.randn(batch_size, 1, 96)

for t in reversed(range(T)):
    noise_pred = model(x_t, t, c_scenario)
    x_t = p_sample(x_t, noise_pred, t)
```

---

# 🔑 10. Practical Recommendations

### Start Here (Strong Baseline Path)

1. Implement **Conv1D cVAE**
2. Validate pipeline end-to-end
3. Then upgrade to diffusion

---

### When to Switch to Diffusion

Switch once you observe:

* Smoothed peaks
* Underestimated variance
* Poor weather response fidelity



# 🧠 Final Insight

You are not just generating synthetic data—you are learning:

[
\textbf{Behavioral demand as a function of physical climate drivers}
]

That’s why:

* Conditioning design matters more than architecture early
* Architecture matters more than conditioning later


Below are **step-by-step, implementation-level instructions** for a coding assistant to build scripts that train the **conditional diffusion model with indoor/outdoor decomposition**, using your existing DuckDB + parquet architecture and respecting:

* **16 GB RAM constraint**
* **10 CPU cores (for preprocessing / dataloading)**
* **Apple GPU via MPS (16-core GPU)**

The instructions assume the repository already contains the structure described in your `README.md`.

---

# 🚀 1. High-Level Pipeline

The system should consist of **four scripts**:

```
scripts/
  01_build_features.R
  02_make_windows.R
  03_export_to_parquet.py
  04_train_diffusion.py
```

Data flow:

```
DuckDB → R (feature engineering) → Parquet (windowed tensors)
       → PyTorch Dataset → Diffusion Training (MPS)
```

---

# 🧱 2. Script 01 — Feature Engineering (R, lazy + parallel)

### Goal

Compute conditioning features **without loading full dataset into memory**

---

## Instructions

1. Connect to DuckDB lazily:

```r
con <- DBI::dbConnect(duckdb::duckdb(), "data/ami_db.duckdb", read_only = TRUE)
panel <- dplyr::tbl(con, "climate_data")
```

---

2. Process **one meter at a time** (CRITICAL for RAM)

* Use `group_split(meter_id)`
* Or better: query meter IDs, loop manually

---

3. For each meter:

* Collect only that meter into memory
* Compute features (lags + rolling)

```r
process_meter <- function(meter_id) {
  df <- panel |>
    filter(meter_id == !!meter_id) |>
    arrange(date_time) |>
    collect()

  df |> build_features()
}
```

---

4. Parallelize across **10 CPU cores**

```r
future::plan(multisession, workers = 10)

results <- future.apply::future_lapply(meter_ids, process_meter)
```

---

5. Write **one parquet file per meter**

```r
arrow::write_parquet(df, paste0("data/features/", meter_id, ".parquet"))
```

---

## Memory Strategy

* Each meter ~140k rows → manageable
* Never load full dataset
* Avoid nested tibbles

---

# 🧱 3. Script 02 — Window Extraction (R → compact tensors)

### Goal

Convert time series into:

* `X`: [N, 96]
* `C_in`: indoor conditioning
* `C_out`: outdoor conditioning

---

## Instructions

1. Read **one meter parquet at a time**

```r
df <- arrow::read_parquet(file)
```

---

2. Construct rolling windows:

```r
make_windows <- function(df) {
  n <- nrow(df)

  purrr::map(1:(n - 95), function(i) {
    list(
      x = df$usage[i:(i+95)],
      c_in = df[i, indoor_cols],
      c_out = df[i, outdoor_cols]
    )
  })
}
```

---

3. Append results to disk **incrementally**

DO NOT accumulate all windows in memory.

Instead:

```r
arrow::write_parquet(batch_df, sink, append = TRUE)
```

---

4. Output schema:

| Column | Shape   |
| ------ | ------- |
| x      | [96]    |
| c_in   | [d_in]  |
| c_out  | [d_out] |

---

## Critical Optimization

* Chunk writing every ~10k windows
* Use Arrow streaming writes

---

# 🧱 4. Script 03 — Export for PyTorch (Python)

### Goal

Convert parquet → memory-efficient PyTorch dataset

---

## Instructions

1. Use **PyArrow Dataset API** (lazy loading)

```python
import pyarrow.dataset as ds

dataset = ds.dataset("data/windows/", format="parquet")
```

---

2. Build iterable dataset (NO full load)

```python
class WaterDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for batch in self.dataset.to_batches(batch_size=1024):
            df = batch.to_pandas()

            x = torch.tensor(df["x"].tolist(), dtype=torch.float32)
            c_in = torch.tensor(df["c_in"].tolist(), dtype=torch.float32)
            c_out = torch.tensor(df["c_out"].tolist(), dtype=torch.float32)

            yield x.unsqueeze(1), c_in, c_out
```

---

## DataLoader Setup

```python
loader = DataLoader(
    dataset,
    batch_size=None,   # already batched
    num_workers=4,     # balance CPU usage
    pin_memory=False   # MPS doesn't benefit
)
```

---

# 🧠 5. Script 04 — Train Diffusion Model

## 5.1 Device Setup (MPS)

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## 5.2 Model: Dual-Stream Diffusion

You must implement:

* Indoor pathway
* Outdoor pathway
* Optional gating

---

### Forward model

[
\epsilon_\theta =
\epsilon^{(in)}(x_t, t, c_{in}) +
\epsilon^{(out)}(x_t, t, c_{out})
]

---

## 5.3 Noise Schedule

Precompute once:

```python
T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
```

---

## 5.4 Forward Diffusion

```python
def q_sample(x0, t, noise):
    sqrt_ab = alpha_bar[t].sqrt().view(-1,1,1)
    sqrt_one_minus = (1 - alpha_bar[t]).sqrt().view(-1,1,1)
    return sqrt_ab * x0 + sqrt_one_minus * noise
```

---

## 5.5 Training Loop (MPS-aware)

```python
for x, c_in, c_out in loader:

    x = x.to(device)
    c_in = c_in.to(device)
    c_out = c_out.to(device)

    noise = torch.randn_like(x)
    t = torch.randint(0, T, (x.size(0),), device=device)

    x_t = q_sample(x, t, noise)

    pred_in, pred_out = model(x_t, t, c_in, c_out)
    noise_pred = pred_in + pred_out

    loss = ((noise - noise_pred) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 5.6 GPU Optimization (IMPORTANT)

For MPS:

* Keep batch size **small (~32–64)**
* Avoid large tensor copies
* Use `.to(device, non_blocking=False)`

---

# ⚙️ 6. Training Configuration

| Component  | Setting     |
| ---------- | ----------- |
| Batch size | 32–64       |
| Epochs     | 50–100      |
| Workers    | 4 CPU       |
| Precision  | float32     |
| Optimizer  | Adam (1e-4) |

---

# 💾 7. Memory Management Strategy

### RAM (16 GB)

* Never load full dataset
* Use:

  * DuckDB (lazy)
  * Arrow streaming
  * IterableDataset

---

### CPU (10 cores)

* 6–8 cores → R preprocessing
* 2–4 cores → PyTorch dataloader

---

### GPU (MPS)

* Only used in training loop
* Avoid preprocessing on GPU

---

# 🧪 8. Sanity Checks (Must Implement)

Before full training:

### 1. Batch Inspection

```python
x.shape == [B, 1, 96]
```

---

### 2. Conditioning Split

* Indoor features contain NO weather
* Outdoor features contain weather

---

### 3. Overfit Small Batch

Train on 1k samples:

* Loss should → near 0

---

# 🔑 9. Final Execution Order

1. Run feature engineering:

```
Rscript scripts/01_build_features.R
```

2. Build windows:

```
Rscript scripts/02_make_windows.R
```

3. Verify parquet output

4. Train model:

```
python scripts/04_train_diffusion.py
```

---

# 🔥 Final Insight

The critical success factors are:

* **Streaming everything** (never load all data)
* **Separating indoor/outdoor conditioning**
* **Using diffusion (not VAE)** to preserve bursts

If any of those break, the synthetic data will look plausible—but fail under scenario testing.
