library(dplyr)
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
  
  # Conditioning: Indoor (c_in) and Outdoor (c_out)
  # Based on build_features script:
  # c_in: month_sin/cos, dow_sin/cos, hour_sin/cos, usage_1h, usage_24h, usage_48h
  c_in_cols <- c("month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos", 
                 "usage_1h", "usage_24h", "usage_48h")
                 
  # c_out: temp_c, precip_mm, snow_cm, temp_1h, temp_24h, temp_48h, snow_flag, precip_3d, snow_24h, gdd_7d
  c_out_cols <- c("temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h", 
                  "snow_flag", "precip_3d", "snow_24h", "gdd_7d")
  
  c_in_mat <- as.matrix(df[, c_in_cols])
  c_out_mat <- as.matrix(df[, c_out_cols])
  
  # Generate indices for rolling windows
  idx <- 1:(n - window_size + 1)
  
  # Map to list of windows
  res <- map(idx, function(i) {
    list(
      x = x_mat[i:(i + window_size - 1), 1],
      c_in = t(c_in_mat[i:(i + window_size - 1), ]), # Transpose to [Channels, Seq]
      c_out = t(c_out_mat[i:(i + window_size - 1), ])
    )
  })
  
  # Flatten and return as a tibble with list-columns
  tibble(
    x = map(res, "x"),
    c_in = map(res, "c_in"),
    c_out = map(res, "c_out")
  )
}

process_windows <- function() {
  # Handle command line arguments for sanity checks
  args <- commandArgs(trailingOnly = TRUE)
  limit_idx <- which(args == "--limit")
  if (length(limit_idx) > 0) {
    n_limit <- as.numeric(args[limit_idx + 1])
    files <- head(files, n_limit)
    cat("Running in SANITY mode: limited to", n_limit, "meters.\n")
  }
  
  cat("Processing", length(files), "files to generate windows...\n")
  
  batch_count <- 0
  current_batch <- list()
  batch_id <- 1
  
  processed_count <- 0
  total_files <- length(files)
  
  for (f in files) {
    df <- arrow::read_parquet(f)
    windows_df <- make_windows(df)
    
    if (!is.null(windows_df) && nrow(windows_df) > 0) {
      current_batch[[length(current_batch) + 1]] <- windows_df
    }
    
    processed_count <- processed_count + 1
    
    # Check if we should write a batch
    if (length(current_batch) >= 50) { # Batched every 50 meters
      batch_df <- bind_rows(current_batch)
      out_path <- file.path(out_dir, paste0("part_", sprintf("%05d", batch_id), ".parquet"))
      arrow::write_parquet(batch_df, out_path)
      
      # Send progress update every batch
      notify_me_done(
        subject = sprintf("📊 Window Extraction Progress: %d/%d files", processed_count, total_files),
        body = sprintf("Just wrote batch %d. Total files processed: %d of %d", batch_id, processed_count, total_files)
      )
      
      batch_id <- batch_id + 1
      current_batch <- list()
    }
  }
  
  # Write any remaining windows
  if (length(current_batch) > 0) {
    batch_df <- bind_rows(current_batch)
    out_path <- file.path(out_dir, paste0("part_", sprintf("%05d", batch_id), ".parquet"))
    arrow::write_parquet(batch_df, out_path)
  }
  
  cat("Window extraction complete. Wrote", batch_id, "parquet files.\n")
}

process_windows()
notify_me_done(subject = "✅ Window extraction finished")
