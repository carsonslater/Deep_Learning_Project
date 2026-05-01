# Enriched Meter-Weather Dataset

This repository contains the pipeline and analytical architecture for generating a high-resolution, enriched panel dataset of AMI water usage and local climate metrics. By the end of the pipeline, thousands of disparate meter records and API-sourced weather observations are synthesized into a unified, queryable structure.

## 📊 The Enriched Data Structure

The primary output of this pipeline is a long-form panel dataset. Each row represents a 15-minute observation for a specific meter, enriched with the local climate conditions at that moment.

### Data Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `meter_id` | `character` | 10-digit unique identifier for the service point. |
| `date_time` | `POSIXct` | Timestamp (15-min intervals) in `America/Denver` timezone. |
| `usage` | `numeric` | Observed water usage for the preceding interval. |
| `temp_c` | `numeric` | Air temperature in Celsius (sourced from Open-Meteo). |
| `precip_mm` | `numeric` | Total precipitation in millimeters. |
| `snow_cm` | `numeric` | Total snowfall in centimeters. |
| `lat` | `numeric` | Service point latitude. |
| `lon` | `numeric` | Service point longitude. |

### Temporal Alignment
Because AMI data is recorded every 15 minutes while archival weather data is typically hourly, the pipeline uses a **DuckDB ASOF JOIN**. Each 15-minute usage record is matched to the nearest *prior* weather observation, ensuring that climate metrics are always available for every usage timestamp without duplicating or averaging raw measurements.

---

## 🗄️ Storage Architecture

To maintain performance across millions of rows, the data is stored in a hybrid architecture:

### 1. Partitioned Parquet Store
Final data is persisted in `data/parquet/` as individual `.parquet` files, one per meter (e.g., `meter_0004100169.parquet`). 
*   **Why**: This allows for rapid parallel writes during the pipeline and enables "predicate pushdown"—software only reads the specific meter files required for a query rather than scanning a massive single file.

### 2. Unified SQL View (`climate_data`)
A virtual VIEW is registered in the persistent DuckDB instance (`data/ami_db.duckdb`) that points to all parquet files simultaneously.
*   **Unified Access**: You can query the entire population as a single table using `SELECT * FROM climate_data`.
*   **Zero-Copy**: The view is a pointer; it does not duplicate the data on disk or in memory.

### 3. Validation Summary (`climate_summary.parquet`)
A high-level health report is generated for the entire dataset, providing per-meter metrics:
*   **Coverage**: Observation counts (`n_obs`) and date ranges (`date_start` to `date_end`).
*   **Data Health**: Percentages of missing values for temperature, precipitation, and usage.
*   **Distribution**: Min/max usage values to identify outliers or flat lines.

---

## 🛠️ Data Access

The data is intended to be accessed lazily to preserve system memory:

```r
library(targets)
library(duckdb)
library(dplyr)

# Connect to the persistent analytical backend
con <- dbConnect(duckdb(), "data/ami_db.duckdb", read_only = TRUE)

# Access the unified panel (0-row load until collect() is called)
panel <- tbl(con, "climate_data")

# Example: Filtering for a specific time/location before pulling into R
high_usage_events <- panel %>%
  filter(temp_c > 30, usage > 10) %>%
  collect()
```

---

## 🧠 Model Architecture: 1D Conditional Diffusion

Beyond the data pipeline, this repository implements a state-of-the-art **1D Conditional Diffusion Model** (based on DDPM) for generating synthetic residential water usage sequences. This model captures both routine human behavior and stochastic, climate-driven irrigation events.

### 1. Zero-Inflated Hurdle Representation
To handle the extreme zero-inflation of residential usage (long periods of $0.0$ flow), the model utilizes a dual-track **Hurdle Transformation**:
*   **Occurrence Mask ($m$)**: A binary channel `[0.0, 1.0]` predicting if flow is occurring.
*   **Log-Magnitude ($v$)**: A continuous channel representing the Z-scored log-volume of the flow.

### 2. Dual-Stream 1D U-Net
The neural backbone is a custom U-Net designed for sequence generation:
*   **Additive Synthesis**: Two parallel streams (`UNet_in` for behavioral/indoor and `UNet_out` for climate/outdoor) are summed to produce the final noise prediction $\epsilon_\theta$.
*   **FiLM Conditioning**: We use **Feature-wise Linear Modulation** to inject complex temporal and weather context (e.g., Growing Degree Days, 6-hour lags) directly into the model's residual blocks.

### 3. Inference & Bernoulli Gating
The model uses a deterministic **Bernoulli Gate** during the reverse diffusion process. By thresholding the predicted mask channel at 0.5, we ensure that generated sequences have perfectly "closed valves" where appropriate, eliminating the low-level noise artifacts typically found in standard diffusion models.

> [!TIP]
> For a deep dive into the mathematical implementation, dual-stream diagrams, and feature engineering details, refer to the [Full Model Architecture Documentation](MODEL_ARCHITECTURE.qmd).

