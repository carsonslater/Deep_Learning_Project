library(dplyr)
library(future)
library(future.apply)
library(arrow)
library(lubridate)
library(tidyr)
library(purrr)
library(zoo)
library(fs)

source("scripts/utils.R")

# Create output directory
dir.create("data/features", recursive = TRUE, showWarnings = FALSE)

# 1. Get meter files from the raw parquet store
meter_files <- fs::dir_ls("data/parquet/", glob = "*.parquet")
meter_files <- meter_files[!stringr::str_detect(meter_files, "climate_summary.parquet")]

# Handle command line arguments for sanity checks
args <- commandArgs(trailingOnly = TRUE)
limit_idx <- which(args == "--limit")
if (length(limit_idx) > 0) {
  n_limit <- as.numeric(args[limit_idx + 1])
  meter_files <- head(meter_files, n_limit)
  cat("Running in SANITY mode: limited to", n_limit, "meters.\n")
}

# Feature engineering function
build_features <- function(df) {
  # Calculate cyclic calendar features
  df <- df %>%
    dplyr::mutate(
      month_t = lubridate::month(date_time),
      dow_t = lubridate::wday(date_time),
      hour_t = lubridate::hour(date_time) + lubridate::minute(date_time) / 60,
      month_sin = sin(2 * pi * month_t / 12),
      month_cos = cos(2 * pi * month_t / 12),
      dow_sin = sin(2 * pi * dow_t / 7),
      dow_cos = cos(2 * pi * dow_t / 7),
      hour_sin = sin(2 * pi * hour_t / 24),
      hour_cos = cos(2 * pi * hour_t / 24)
    )

  # Calculate weather memory features
  df <- df %>%
    dplyr::arrange(date_time) %>%
    dplyr::mutate(
      # Temperature lags (1h=4 steps, 24h=96 steps, 48h=192 steps)
      temp_1h = dplyr::lag(temp_c, 4),
      temp_24h = dplyr::lag(temp_c, 96),
      temp_48h = dplyr::lag(temp_c, 192),

      # Snow flag
      snow_flag = dplyr::if_else(snow_cm > 0, 1, 0)
    )

  df <- df %>%
    dplyr::mutate(
      precip_3d = zoo::rollapplyr(precip_mm, width = 288, FUN = sum, fill = NA, partial = TRUE),
      snow_24h = zoo::rollapplyr(snow_cm, width = 96, FUN = sum, fill = NA, partial = TRUE),

      # GDD (Growing Degree Days style transform: max(temp - 10, 0))
      gdd_base = pmax(temp_c - 10, 0),
      gdd_7d = zoo::rollapplyr(gdd_base, width = 672, FUN = sum, fill = NA, partial = TRUE)
    )

  # Lagged usage statistics
  df <- df %>%
    dplyr::mutate(
      usage_15m = dplyr::lag(usage, 1),
      usage_30m = dplyr::lag(usage, 2),
      usage_1h = dplyr::lag(usage, 4),
      usage_6h = dplyr::lag(usage, 24),
      usage_24h = dplyr::lag(usage, 96),
      usage_48h = dplyr::lag(usage, 192)
    )

  # Clean up temporary columns
  df <- df %>%
    dplyr::select(-month_t, -dow_t, -hour_t, -gdd_base) %>%
    tidyr::drop_na() # Drop initial rows that don't have enough history for lags

  return(df)
}

# 2. Process meter function
process_meter_file <- function(f) {
  fname <- fs::path_file(f)
  mid <- stringr::str_extract(fname, "(?<=meter_)[0-9]+")

  df <- arrow::read_parquet(f)

  # Filter for 2023 and 2024 data
  df <- df %>% dplyr::filter(lubridate::year(date_time) %in% c(2023, 2024))

  if (nrow(df) > 0) {
    df_features <- build_features(df)
    arrow::write_parquet(df_features, paste0("data/features/meter_", mid, ".parquet"))
  }

  return(TRUE)
}

# 3. Set up parallel plan
n_workers <- max(1, parallelly::availableCores() - 1)
cat("Utilizing", n_workers, "cores for feature generation.\n")
future::plan(multisession, workers = n_workers)

# 4. Run extraction with progress tracking
cat("Starting feature generation for", length(meter_files), "files...\n")

total_files <- length(meter_files)
chunk_size <- max(1, floor(total_files / 10)) # Update every 10%

for (i in seq(1, total_files, by = chunk_size)) {
  end_idx <- min(i + chunk_size - 1, total_files)
  current_chunk <- meter_files[i:end_idx]

  results <- future.apply::future_lapply(current_chunk, future.scheduling = FALSE, function(f) {
    # ANTI-THRASHING: Force workers to be single-threaded
    Sys.setenv(OMP_NUM_THREADS = "1")
    Sys.setenv(OPENBLAS_NUM_THREADS = "1")
    Sys.setenv(MKL_NUM_THREADS = "1")
    if (requireNamespace("arrow", quietly = TRUE)) {
      arrow::set_cpu_count(1)
    }
    process_meter_file(f)
  })

  # Only send a notification around the 50% mark
  progress_pct <- round(end_idx / total_files * 100)
  if (progress_pct >= 50 && (progress_pct - round(length(current_chunk) / total_files * 100)) < 50) {
    notify_me_done(
      subject = sprintf("[STATUS] Feature Generation Progress: %d%%", progress_pct),
      body = sprintf("Completed %d of %d files.", end_idx, total_files)
    )
  }
}

cat("Completed feature generation.\n")
notify_me_done(subject = "[DONE] Feature generation finished")
