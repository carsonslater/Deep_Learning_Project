# 08_evaluate_dtw.R

library(tidyverse)
library(dtw)
library(dtwclust)
library(proxy)

#' 5. Advanced Dynamic Time Warping (DTW) & Structural Alignment
#'
#' @param data A combined dataset of Real and Synthetic sequences.
#' @param window_size Sakoe-Chiba band window size.
#' @return A list containing the DBA archetypes, alignment object, summary table, and visualization.
evaluate_dtw_alignment <- function(data, window_size = 8) {
  
  # Extract and format sequences into lists of numeric vectors
  real_seqs <- data |>
    dplyr::filter(source == "Real") |>
    dplyr::arrange(home_id, time) |>
    dplyr::group_split(home_id) |>
    purrr::map(~ .x$usage)
  
  synth_seqs <- data |>
    dplyr::filter(source == "Synthetic") |>
    dplyr::arrange(home_id, time) |>
    dplyr::group_split(home_id) |>
    purrr::map(~ .x$usage)
  
  if (length(real_seqs) == 0 || length(synth_seqs) == 0) {
    stop("Insufficient data for DTW evaluation. Check your data filtering.")
  }
  
  # ---------------------------------------------------------
  # 5.1 DTW Barycenter Averaging (DBA)
  # ---------------------------------------------------------
  # Compute true time-series archetype (avoids flattening phase-shifted peaks)
  real_dba <- dtwclust::dba(real_seqs)
  synth_dba <- dtwclust::dba(synth_seqs)
  
  # ---------------------------------------------------------
  # 5.2 Constrained Alignment
  # ---------------------------------------------------------
  # Calculate DTW distance between Archetypes with Sakoe-Chiba band
  alignment <- dtw::dtw(
    x = real_dba, 
    y = synth_dba, 
    window.type = "sakoechiba", 
    window.size = window_size
  )
  
  # ---------------------------------------------------------
  # 5.3 Distributional DTW Assessment
  # ---------------------------------------------------------
  # Sample 100 individual sequences from both
  n_sample <- min(100, length(real_seqs), length(synth_seqs))
  real_sample <- sample(real_seqs, n_sample)
  synth_sample <- sample(synth_seqs, n_sample)
  
  # To avoid extremely long R-level loops, sample 1000 pairs for distributions
  n_pairs <- 1000
  rr_dists <- numeric(n_pairs)
  ss_dists <- numeric(n_pairs)
  rs_dists <- numeric(n_pairs)
  
  set.seed(42)
  for (i in seq_len(n_pairs)) {
    # Real vs Real
    i1 <- sample(n_sample, 1); i2 <- sample(n_sample, 1)
    rr_dists[i] <- dtw::dtw(
      real_sample[[i1]], real_sample[[i2]], 
      distance.only = TRUE, window.type = "sakoechiba", window.size = window_size
    )$distance
    
    # Synthetic vs Synthetic
    i1 <- sample(n_sample, 1); i2 <- sample(n_sample, 1)
    ss_dists[i] <- dtw::dtw(
      synth_sample[[i1]], synth_sample[[i2]], 
      distance.only = TRUE, window.type = "sakoechiba", window.size = window_size
    )$distance
    
    # Real vs Synthetic
    i1 <- sample(n_sample, 1); i2 <- sample(n_sample, 1)
    rs_dists[i] <- dtw::dtw(
      real_sample[[i1]], synth_sample[[i2]], 
      distance.only = TRUE, window.type = "sakoechiba", window.size = window_size
    )$distance
  }
  
  # Output Summary Table
  dist_summary <- tibble::tibble(
    Comparison = c("Real-vs-Real", "Synthetic-vs-Synthetic", "Real-vs-Synthetic"),
    Mean_DTW_Distance = c(mean(rr_dists), mean(ss_dists), mean(rs_dists)),
    Variance_DTW_Distance = c(var(rr_dists), var(ss_dists), var(rs_dists))
  )
  
  # ---------------------------------------------------------
  # 5.4 Native ggplot2 Visualization
  # ---------------------------------------------------------
  # Extract alignment path vectors directly
  path_x <- alignment$index1
  path_y <- alignment$index2
  
  # Offset the Synthetic archetype by a fixed amount for visual separation
  offset_val <- max(real_dba) * 1.5 
  t_index <- seq_along(real_dba)
  
  # Dataframes for the archetype curves
  curves_df <- dplyr::bind_rows(
    tibble::tibble(time = t_index, value = real_dba, source = "Real Archetype"),
    tibble::tibble(time = t_index, value = synth_dba + offset_val, source = "Synthetic Archetype")
  ) |>
    dplyr::mutate(source = factor(source, levels = c("Real Archetype", "Synthetic Archetype")))
  
  # Dataframe for the alignment mapping lines
  mapping_df <- tibble::tibble(
    x_start = path_x,
    x_end = path_y,
    y_start = real_dba[path_x],
    y_end = synth_dba[path_y] + offset_val
  )
  
  # Downsample mapping lines for visual clarity (e.g., plot every 2nd line)
  mapping_df_plot <- mapping_df |>
    dplyr::filter(dplyr::row_number() %% 2 == 0)
  
  dtw_plot <- ggplot2::ggplot() +
    # Draw mapping segments
    ggplot2::geom_segment(
      data = mapping_df_plot,
      ggplot2::aes(x = x_start, y = y_start, xend = x_end, yend = y_end),
      color = "gray60", linetype = "dotted", alpha = 0.6
    ) +
    # Draw the two archetype curves
    ggplot2::geom_line(
      data = curves_df,
      ggplot2::aes(x = time, y = value, color = source),
      linewidth = 1.2
    ) +
    ggplot2::scale_color_manual(
      values = c("Real Archetype" = "#1b9e77", "Synthetic Archetype" = "#d95f02")
    ) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "DTW Structural Alignment (DBA Archetypes)",
      subtitle = sprintf("Sakoe-Chiba Band (Window = %d) | Archetype DTW Distance: %.2f", 
                         window_size, alignment$distance),
      x = "Time Index (15-min Intervals)",
      y = "Usage Magnitude (Offset Applied for Clarity)",
      color = "Source"
    ) +
    ggplot2::theme(
      axis.text.y = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank(),
      legend.position = "bottom"
    )
  
  return(list(
    summary_table = dist_summary,
    plot = dtw_plot
  ))
}

# ------------------------------------------------------------------------------
# Process 500 Synthetic and 500 Real sequences for each month of the year
# ------------------------------------------------------------------------------

run_monthly_dtw_pipeline <- function(combined_dataset) {
  # Add month column if not already present
  if (!"month" %in% colnames(combined_dataset)) {
    combined_dataset <- combined_dataset |>
      dplyr::mutate(month = lubridate::month(time, label = TRUE, abbr = TRUE))
  }
  
  months <- unique(combined_dataset$month)
  results_by_month <- list()
  
  for (m in months) {
    cat(sprintf("\n--- Executing DTW Pipeline for Month: %s ---\n", m))
    
    # Filter for specific month
    m_data <- combined_dataset |>
      dplyr::filter(month == m)
    
    # Sample 500 homes from Real and 500 from Synthetic
    real_homes <- m_data |> dplyr::filter(source == "Real") |> dplyr::pull(home_id) |> unique()
    synth_homes <- m_data |> dplyr::filter(source == "Synthetic") |> dplyr::pull(home_id) |> unique()
    
    n_real <- min(500, length(real_homes))
    n_synth <- min(500, length(synth_homes))
    
    set.seed(m)
    selected_real <- sample(real_homes, n_real)
    selected_synth <- sample(synth_homes, n_synth)
    
    sampled_m_data <- m_data |>
      dplyr::filter(
        (source == "Real" & home_id %in% selected_real) |
        (source == "Synthetic" & home_id %in% selected_synth)
      )
    
    # Run evaluation
    res <- evaluate_dtw_alignment(sampled_m_data, window_size = 8)
    results_by_month[[as.character(m)]] <- res
    
    # Print results
    cat("Distributional Summary Table:\n")
    print(res$summary_table)
    
    # Print the plot (or you can save it)
    print(res$plot)
  }
  
  return(results_by_month)
}

# # Usage Example:
# # Assuming `combined_annual_data` is a dataset containing a full year of 
# # both real and synthetic data traces.
# # results <- run_monthly_dtw_pipeline(combined_annual_data)
