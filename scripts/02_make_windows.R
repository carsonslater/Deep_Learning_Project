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
  # Handle command line arguments for sanity checks
  files <- fs::dir_ls(in_dir, glob = "*.parquet")
  
  args <- commandArgs(trailingOnly = TRUE)
  limit_idx <- which(args == "--limit")
  if (length(limit_idx) > 0) {
    n_limit <- as.numeric(args[limit_idx + 1])
    files <- head(files, n_limit)
    cat("Running in SANITY mode: limited to", n_limit, "meters.\n")
  }
  
  cat("Processing", length(files), "files to generate windows (Disjoint Mode)...\n")
  
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
    
    # Notify every 200 homes
    if (processed_count %% 200 == 0) {
      notify_me_done(
        subject = sprintf("🏠 Window Extraction: %d/%d Homes Complete", processed_count, total_files),
        body = sprintf("Progress: %.1f%%", (processed_count / total_files) * 100)
      )
      # Trigger GC when notifying
      gc()
    }
    
    # Write batches every 100 homes (slightly larger batches for disjoint mode)
    if (length(current_batch) >= 100) {
      batch_df <- bind_rows(current_batch)
      out_path <- file.path(out_dir, paste0("part_", sprintf("%05d", batch_id), ".parquet"))
      arrow::write_parquet(batch_df, out_path)
      
      batch_id <- batch_id + 1
      current_batch <- list()
      # Periodic GC
      gc()
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
notify_me_done(subject = "✅ Disjoint Window extraction finished")
