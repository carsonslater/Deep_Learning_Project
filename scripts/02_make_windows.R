library(arrow)
library(dplyr)
library(purrr)
library(fs)

# Directories
in_dir <- "data/features"
out_dir <- "data/windows"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# We want an indoor and outdoor conditioning split
# c_in: calendar encodings, lagged usage
# c_out: temp, precip, snow, lagged weather
indoor_cols <- c(
  "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos",
  "usage_1h", "usage_24h", "usage_48h"
)

outdoor_cols <- c(
  "temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h",
  "snow_flag", "precip_3d", "snow_24h", "gdd_7d"
)

make_windows <- function(df) {
  n <- nrow(df)
  if (n <= 96) return(NULL)
  
  # We extract windows of length 96 for x. 
  # For c_in and c_out, we use the conditioning variables at the start of the window, 
  # or at the end of the window. Let's use the features corresponding to the end of the window (i+95).
  # Wait, usually predicting sequence based on current day's conditioning. Let's use i for simplicity
  # as per the example in HOW_TO_IMPLEMENT.md:
  # list(x = df$usage[i:(i+95)], c_in = df[i, indoor_cols], c_out = df[i, outdoor_cols])
  
  res <- purrr::map(1:(n - 95), function(i) {
    list(
      x = as.numeric(unlist(df$usage[i:(i+95)])),
      c_in = as.numeric(unlist(df[i, indoor_cols])),
      c_out = as.numeric(unlist(df[i, outdoor_cols]))
    )
  })
  
  # Convert list of lists to dataframe
  tibble(
    x = map(res, "x"),
    c_in = map(res, "c_in"),
    c_out = map(res, "c_out")
  )
}

process_windows <- function() {
  files <- fs::dir_ls(in_dir, glob = "*.parquet")
  files <- head(files, 10)
  
  cat("Processing", length(files), "files to generate windows...\n")
  
  batch_count <- 0
  current_batch <- list()
  batch_id <- 1
  
  for (f in files) {
    df <- arrow::read_parquet(f)
    windows_df <- make_windows(df)
    
    if (!is.null(windows_df) && nrow(windows_df) > 0) {
      current_batch[[length(current_batch) + 1]] <- windows_df
    }
    
    # Check if we should write a batch (e.g., every 5 meters to keep chunk size ~50k windows)
    # A single meter has ~140k/4/24 ~ maybe many windows. Let's just write meter by meter, 
    # but append it to a dataset directory with partitioning or just individual files
    
    if (length(current_batch) >= 5) {
      batch_df <- bind_rows(current_batch)
      
      # We write it as a part of a dataset
      out_path <- file.path(out_dir, paste0("part_", sprintf("%05d", batch_id), ".parquet"))
      arrow::write_parquet(batch_df, out_path)
      
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
