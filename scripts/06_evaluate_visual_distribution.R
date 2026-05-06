# 06_evaluate_visual_distribution.R

library(tidyverse)
library(transport) # For Wasserstein distance

#' Compile Real Data
#'
#' @param directory_path Path to directory with real data files
#' @param target_doy Target day of year to filter by
#' @param n_homes Number of homes to sample
#' @return A tidy dataframe of real usage
compile_real_data <- function(directory_path, target_doy, n_homes) {
  # Detect file types
  files <- list.files(directory_path, pattern = "\\.(csv|parquet)$", full.names = TRUE)

  if (length(files) == 0) {
    stop("No data files found in the specified directory.")
  }

  # Read and combine data. Assumes columns `doy`, `home_id`, `time`, and `usage`.
  if (grepl("\\.parquet$", files[1])) {
    data_list <- purrr::map(files, arrow::read_parquet)
  } else {
    data_list <- purrr::map(files, readr::read_csv)
  }

  combined_data <- purrr::list_rbind(data_list) |>
    dplyr::filter(doy == target_doy)

  # Sample a subset of homes
  unique_homes <- unique(combined_data$home_id)
  if (length(unique_homes) < n_homes) {
    warning("Fewer homes available than n_homes requested. Using all available homes.")
    selected_homes <- unique_homes
  } else {
    selected_homes <- sample(unique_homes, n_homes)
  }

  final_data <- combined_data |>
    dplyr::filter(home_id %in% selected_homes) |>
    dplyr::mutate(source = "Real")

  return(final_data)
}

#' Generate Synthetic Data (Updated to reflect conditional generation)
#'
#' @param real_covariates A tibble containing the condition variables (c_in, c_out) from real data
#' @param n Number of samples (homes) to generate for these conditions
#' @return A tibble of synthetic traces matching real data structure
generate_diffusion_samples <- function(real_covariates, n) {
  # Call Python diffusion model or simulate matching traces
  
  # Simulate samples matching covariates
  
  daily_temp_profile <- real_covariates |>
    dplyr::group_by(time) |>
    dplyr::summarize(temp_c = mean(temp_c, na.rm = TRUE), .groups = "drop")
  
  intervals <- sort(unique(real_covariates$time))
  
  # Simulate right-skewed data that responds to temperature (more heat = more usage)
  synthetic_data <- purrr::map_dfr(1:n, function(i) {
    # Base usage + temperature response
    temp_effect <- (daily_temp_profile$temp_c - 20) * 0.05
    usage <- rlnorm(96, meanlog = -1 + pmax(0, temp_effect), sdlog = 1)
    
    # Inject zeros
    zeros <- runif(96) < 0.6
    usage[zeros] <- 0
    
    tibble::tibble(
      home_id = paste0("synth_", i),
      time = intervals,
      usage = usage,
      source = "Synthetic"
    )
  })
  
  return(synthetic_data)
}

# ------------------------------------------------------------------------------
# Workflow for Visual & Distributional Assessment
# ------------------------------------------------------------------------------

assess_distributions <- function(real_data, synthetic_data) {
  # Combine data and ensure "Real" is on the left
  combined_data <- dplyr::bind_rows(real_data, synthetic_data) |>
    dplyr::mutate(
      source = factor(source, levels = c("Real", "Synthetic")),
      # Extract time-of-day for alignment
      time_of_day = as.POSIXct(format(time, "%H:%M:%S"), format = "%H:%M:%S", tz = "UTC")
    )

  mean_usage <- combined_data |>
    dplyr::group_by(source, time_of_day) |>
    dplyr::summarize(mean_usage = mean(usage, na.rm = TRUE), .groups = "drop")

  trace_plot <- ggplot2::ggplot(combined_data, ggplot2::aes(x = time_of_day, y = usage, group = home_id, color = source)) +
    ggplot2::geom_line(alpha = 0.05) +
    ggplot2::geom_line(
      data = mean_usage,
      ggplot2::aes(x = time_of_day, y = mean_usage, group = source, color = source),
      linewidth = 1.2,
      linetype = "dashed"
    ) +
    ggplot2::scale_x_datetime(date_labels = "%H:%M", date_breaks = "4 hours") +
    ggplot2::facet_wrap(~source, ncol = 2) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "24-Hour Residential Water Usage Traces",
      subtitle = "Individual traces (solid, low alpha) with mean usage (dashed)",
      x = "Time of Day",
      y = "Water Usage",
      color = "Source"
    ) +
    ggplot2::theme(legend.position = "none")

  print(trace_plot)

  non_zero_data <- combined_data |>
    dplyr::filter(usage > 0)

  real_non_zero <- non_zero_data |>
    dplyr::filter(source == "Real") |>
    dplyr::pull(usage)
  synth_non_zero <- non_zero_data |>
    dplyr::filter(source == "Synthetic") |>
    dplyr::pull(usage)

  # Calculate 1-Wasserstein Distance
  w_dist <- transport::wasserstein1d(real_non_zero, synth_non_zero)

  # Density Plot
  density_plot <- ggplot2::ggplot(non_zero_data, ggplot2::aes(x = usage, fill = source)) +
    ggplot2::geom_density(alpha = 0.5) +
    ggplot2::scale_x_log10() + # Log scale is recommended due to right-skewness
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Non-Zero Water Usage Magnitudes",
      subtitle = sprintf("1-Wasserstein Distance: %.4f", w_dist),
      x = "Water Usage (Log Scale)",
      y = "Density",
      fill = "Source"
    )

  print(density_plot)
}
