library(dplyr)
library(future)
library(future.apply)
library(arrow)
library(purrr)
library(fs)

source("scripts/utils.R")

# Directories
in_dir <- "data/features"
out_dir <- "data/windows"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Window extraction function
make_windows <- function(df, window_size = 96) {
  n <- nrow(df)
  if (n < window_size) return(NULL)
  
  # Target sequence: usage (x)
  x_mat <- as.matrix(df$usage)
  
  # c_in: month_sin/cos, dow_sin/cos, hour_sin/cos, usage_1h, usage_24h, usage_48h
  c_in_cols <- c("month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos", 
                 "usage_1h", "usage_6h", "usage_24h", "usage_48h")
                 
  # c_out: temp_c, precip_mm, snow_cm, temp_1h, temp_24h, temp_48h, snow_flag, precip_3d, snow_24h, gdd_7d
  c_out_cols <- c("temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h", 
                  "snow_flag", "precip_3d", "snow_24h", "gdd_7d")
  
  c_in_mat <- as.matrix(df[, c_in_cols])
  c_out_mat <- as.matrix(df[, c_out_cols])
  
  # Use rolling windows (stride = 1) for data augmentation and translation invariance
  # This provides ~96x more training data.
  stride <- 1 
  idx <- seq(1, n - window_size + 1, by = stride)
  
  # Optimized extraction: Pre-slice matrices
  res <- lapply(idx, function(i) {
    list(
      x = x_mat[i:(i + window_size - 1), 1],
      c_in = t(c_in_mat[i:(i + window_size - 1), ]),
      c_out = t(c_out_mat[i:(i + window_size - 1), ])
    )
  })
  
  # Return as a tibble
  tibble(
    x = lapply(res, `[[`, "x"),
    c_in = lapply(res, `[[`, "c_in"),
    c_out = lapply(res, `[[`, "c_out")
  )
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
  
  cat("Processing", length(files), "files to generate windows (Parallel Mode)...\n")
  
  # Parallel Setup - use all available cores
  n_workers <- parallelly::availableCores()
  cat("Utilizing", n_workers, "cores.\n")
  future::plan(multisession, workers = n_workers)
  
  total_files <- length(files)
  chunk_size <- 100 # Process 100 homes at a time in parallel to manage memory
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
    
    # Filter out NULLs and bind
    batch_df <- bind_rows(chunk_results)
    
    if (nrow(batch_df) > 0) {
      out_path <- file.path(out_dir, paste0("part_", sprintf("%05d", batch_id), ".parquet"))
      arrow::write_parquet(batch_df, out_path)
      batch_id <- batch_id + 1
    }
    
    cat(sprintf("Chunk %d complete (Homes %d-%d)\n", floor(i/chunk_size) + 1, i, end_idx))
    
    # Progress notification every 500 homes
    if (i %% 500 == 1 && i > 1) {
      notify_me_done(
        subject = sprintf("🏠 Window Extraction Progress: %d%%", round(end_idx / total_files * 100)),
        body = sprintf("Completed %d of %d files.", end_idx, total_files)
      )
    }
    
    # Explicit clean up in main process
    rm(chunk_results, batch_df)
    gc(full = TRUE)
  }

  
  cat("Window extraction complete. Wrote", batch_id - 1, "parquet files.\n")
}

process_windows()
notify_me_done(subject = "✅ Window extraction finished (Parallel)")

