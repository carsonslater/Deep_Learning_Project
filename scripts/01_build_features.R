library(dplyr)
library(duckdb)
library(future)
library(future.apply)
library(arrow)
library(lubridate)
library(tidyr)
library(purrr)

# 1. Connect to DuckDB lazily
con <- DBI::dbConnect(duckdb::duckdb(), "data/ami_db.duckdb", read_only = TRUE)
panel <- dplyr::tbl(con, "climate_data")

# Create output directory
dir.create("data/features", recursive = TRUE, showWarnings = FALSE)

# 2. Get meter IDs
# We only want to enumerate meter IDs first to process them one at a time
meter_ids <- panel %>%
  distinct(meter_id) %>%
  collect() %>%
  pull(meter_id)

meter_ids <- head(meter_ids, 10)

# Feature engineering function
build_features <- function(df) {
  # Calculate cyclic calendar features
  df <- df %>%
    mutate(
      month_t = month(date_time),
      dow_t = wday(date_time),
      hour_t = hour(date_time) + minute(date_time) / 60,
      month_sin = sin(2 * pi * month_t / 12),
      month_cos = cos(2 * pi * month_t / 12),
      dow_sin = sin(2 * pi * dow_t / 7),
      dow_cos = cos(2 * pi * dow_t / 7),
      hour_sin = sin(2 * pi * hour_t / 24),
      hour_cos = cos(2 * pi * hour_t / 24)
    )

  # Calculate weather memory features
  df <- df %>%
    arrange(date_time) %>%
    mutate(
      # Temperature lags (1h=4 steps, 24h=96 steps, 48h=192 steps)
      temp_1h = lag(temp_c, 4),
      temp_24h = lag(temp_c, 96),
      temp_48h = lag(temp_c, 192),

      # Snow flag
      snow_flag = if_else(snow_cm > 0, 1, 0)
    )

  # Rolling features using Slider or equivalent could be memory intensive,
  # but here we are meter-by-meter so it's okay. Using simple zoo::rollsum or just keep it simple.
  # For 3d precip (288 steps), 24h snow (96 steps), 7d GDD (672 steps)

  library(zoo)
  df <- df %>%
    mutate(
      precip_3d = rollapplyr(precip_mm, width = 288, FUN = sum, fill = NA, partial = TRUE),
      snow_24h = rollapplyr(snow_cm, width = 96, FUN = sum, fill = NA, partial = TRUE),

      # GDD (Growing Degree Days style transform: max(temp - 10, 0))
      gdd_base = pmax(temp_c - 10, 0),
      gdd_7d = rollapplyr(gdd_base, width = 672, FUN = sum, fill = NA, partial = TRUE)
    )

  # Lagged usage statistics
  df <- df %>%
    mutate(
      usage_1h = lag(usage, 4),
      usage_24h = lag(usage, 96),
      usage_48h = lag(usage, 192)
    )

  # Clean up temporary columns
  df <- df %>%
    select(-month_t, -dow_t, -hour_t, -gdd_base) %>%
    drop_na() # Drop initial rows that don't have enough history for lags

  return(df)
}

# 3. Process meter function
process_meter <- function(mid) {
  # Re-connect to DuckDB inside the worker to avoid connection sharing issues across processes
  worker_con <- DBI::dbConnect(duckdb::duckdb(), "data/ami_db.duckdb", read_only = TRUE)
  worker_panel <- dplyr::tbl(worker_con, "climate_data")

  df <- worker_panel %>%
    filter(meter_id == !!mid) %>%
    arrange(date_time) %>%
    collect()

  DBI::dbDisconnect(worker_con)

  if (nrow(df) > 0) {
    df_features <- build_features(df)
    arrow::write_parquet(df_features, paste0("data/features/meter_", mid, ".parquet"))
  }

  return(TRUE)
}

# 4. Parallelize across 10 CPU cores
# Set up parallel plan
future::plan(multisession, workers = 10)

# Run extraction
cat("Starting feature generation for", length(meter_ids), "meters...\n")
results <- future.apply::future_lapply(meter_ids, process_meter)
cat("Completed feature generation.\n")

DBI::dbDisconnect(con)
