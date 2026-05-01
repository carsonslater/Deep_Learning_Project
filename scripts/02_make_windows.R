library(data.table)
library(future)
library(future.apply)
library(arrow)
library(fs)

source("scripts/utils.R")

# Directories
in_dir <- "data/features"
out_dir <- "data/windows"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Window extraction function optimized with data.table
make_windows <- function(df, window_size = 96) {
  n <- nrow(df)
  if (n < window_size) {
    return(NULL)
  }

  # Convert to data.table in-place to avoid copying
  data.table::setDT(df)

  # c_in: month_sin/cos, dow_sin/cos, hour_sin/cos, usage_1h, usage_24h, usage_48h
  c_in_cols <- c(
    "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos",
    "usage_15m", "usage_30m", "usage_1h", "usage_6h", "usage_24h", "usage_48h"
  )

  # c_out: temp_c, precip_mm, snow_cm, temp_1h, temp_24h, temp_48h, snow_flag, precip_3d, snow_24h, gdd_7d
  c_out_cols <- c(
    "temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h",
    "snow_flag", "precip_3d", "snow_24h", "gdd_7d"
  )

  # Pre-extract vectors and matrices for fast slicing
  x_vec <- df$usage
  time_vec <- df$date_time
  c_in_mat <- as.matrix(df[, ..c_in_cols])
  c_out_mat <- as.matrix(df[, ..c_out_cols])

  # Use rolling windows (stride = 1) for data augmentation and translation invariance
  stride <- 1
  raw_idx <- seq(1, n - window_size + 1, by = stride)

  # [CRITICAL FIX] Ensure Windows are Temporally Continuous
  # Since interval is 15 mins, a 96-step window MUST span exactly 1425 minutes (95 * 15)
  # If it spans more, it means there is a missing data gap spliced inside it.
  time_diffs <- difftime(time_vec[raw_idx + window_size - 1], time_vec[raw_idx], units = "mins")
  idx <- raw_idx[time_diffs == 1425]

  if (length(idx) == 0) {
    return(NULL)
  }

  # Return as a data.table with list-columns
  dt_res <- data.table::data.table(
    x = lapply(idx, function(i) x_vec[i:(i + window_size - 1)]),
    c_in = lapply(idx, function(i) t(c_in_mat[i:(i + window_size - 1), ])),
    c_out = lapply(idx, function(i) t(c_out_mat[i:(i + window_size - 1), ]))
  )

  return(dt_res)
}

process_windows <- function() {
  files <- fs::dir_ls(in_dir, glob = "*.parquet")

  args <- commandArgs(trailingOnly = TRUE)
  limit_idx <- which(args == "--limit")
  if (length(limit_idx) > 0) {
    n_limit <- as.numeric(args[limit_idx + 1])
    files <- head(files, n_limit)
    cat("Running in SANITY mode: limited to", n_limit, "meters.\n")
  }

  cat("Processing", length(files), "files to generate windows (Parallel Mode with data.table)...\n")

  # Parallel Setup - use all available cores
  n_workers <- parallelly::availableCores()
  cat("Utilizing", n_workers, "cores.\n")
  future::plan(multisession, workers = n_workers)

  total_files <- length(files)
  chunk_size <- 10 # Drastically reduced to prevent 30GB memory spikes with stride=1
  batch_id <- 1

  for (i in seq(1, total_files, by = chunk_size)) {
    end_idx <- min(i + chunk_size - 1, total_files)
    current_chunk <- files[i:end_idx]

    # Process chunk in parallel
    chunk_results <- future.apply::future_lapply(current_chunk, function(f) {
      df <- arrow::read_parquet(f)
      res <- make_windows(df)
      # Clean up worker memory after processing large window matrices
      rm(df)
      gc(full = TRUE)
      return(res)
    })

    # rbindlist is exponentially faster than dplyr::bind_rows for list columns
    batch_dt <- data.table::rbindlist(chunk_results)

    if (nrow(batch_dt) > 0) {
      out_path <- file.path(out_dir, paste0("part_", sprintf("%05d", batch_id), ".parquet"))
      arrow::write_parquet(batch_dt, out_path)
      batch_id <- batch_id + 1
    }

    cat(sprintf("Chunk %d complete (Homes %d-%d)\n", floor(i / chunk_size) + 1, i, end_idx))

    # Only send a notification around the 50% mark
    progress_pct <- round(end_idx / total_files * 100)
    if (progress_pct >= 50 && (progress_pct - round(length(current_chunk) / total_files * 100)) < 50) {
      notify_me_done(
        subject = sprintf("[STATUS] Window Extraction Progress: %d%%", progress_pct),
        body = sprintf("Completed %d of %d files.", end_idx, total_files)
      )
    }

    # Explicit clean up in main process
    rm(chunk_results, batch_dt)
    gc(full = TRUE)
  }

  cat("Window extraction complete. Wrote", batch_id - 1, "parquet files.\n")
}

process_windows()
notify_me_done(subject = "[DONE] Window extraction finished (data.table Parallel)")
