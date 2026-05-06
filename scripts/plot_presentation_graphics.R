library(tidyverse)
library(patchwork)

# 1. Plot Synthetic Samples POC
if (file.exists("scratch/synthetic_samples_raw.csv")) {
  samples_df <- read_csv("scratch/synthetic_samples_raw.csv")
  
  # Calculate mean profile
  mean_profile <- samples_df %>%
    group_by(interval) %>%
    summarize(usage = mean(usage), .groups = "drop")
  
  p1 <- ggplot(samples_df, aes(x = interval, y = usage, group = sample_id)) +
    geom_line(alpha = 0.2, color = "#1abc9c") +
    geom_line(data = mean_profile, aes(x = interval, y = usage, group = 1), 
              color = "white", linewidth = 1.5, linetype = "dashed") +
    labs(
      title = "Synthetic Water Usage Generation (Diffusion Model POC)",
      subtitle = "50 synthetic traces generated on MPS (Optimized T=0.8)",
      x = "Time of Day (15-min Intervals)",
      y = "Usage Intensity (Gallons)"
    ) +
    theme_dark() +
    theme(
      plot.background = element_rect(fill = "#0a0a0a", color = NA),
      panel.background = element_rect(fill = "#0a0a0a", color = NA),
      panel.grid.major = element_line(color = "#222222"),
      panel.grid.minor = element_blank(),
      text = element_text(color = "#cccccc"),
      plot.title = element_text(color = "white", face = "bold", size = 18),
      axis.text = element_text(color = "#888888")
    )
  
  ggsave("images/synthetic_samples_poc_ggplot.png", p1, width = 12, height = 6, dpi = 150)
  cat("Saved images/synthetic_samples_poc_ggplot.png\n")
}

# 2. Plot Optimized Distribution
if (file.exists("scratch/distribution_data.csv") && file.exists("scratch/profile_data.csv")) {
  dist_df <- read_csv("scratch/distribution_data.csv")
  profile_df <- read_csv("scratch/profile_data.csv")
  
  # Profile Comparison
  profile_long <- profile_df %>%
    pivot_longer(cols = c(real_mean, synthetic_mean), names_to = "source", values_to = "usage") %>%
    mutate(source = if_else(source == "real_mean", "Real", "Synthetic"))
  
  p_profile <- ggplot(profile_long, aes(x = interval, y = usage, color = source, linetype = source)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = c("Real" = "#3498db", "Synthetic" = "#2ecc71")) +
    labs(title = "Mean Usage Profile Comparison", x = "Interval", y = "Gallons") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Distribution Comparison (log1p)
  p_dist <- ggplot(dist_df, aes(x = usage, fill = source)) +
    geom_density(alpha = 0.5) +
    scale_x_log10() +
    scale_fill_manual(values = c("Real" = "#3498db", "Synthetic" = "#2ecc71")) +
    labs(title = "Magnitude Distribution (Log Scale)", x = "Usage (log10 scale)", y = "Density") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  combined_eval <- p_profile + p_dist + 
    plot_annotation(title = "Inference Optimization Results", theme = theme(plot.title = element_text(size = 18, face = "bold")))
  
  ggsave("images/optimized_distribution_ggplot.png", combined_eval, width = 14, height = 6, dpi = 150)
  cat("Saved images/optimized_distribution_ggplot.png\n")
}
