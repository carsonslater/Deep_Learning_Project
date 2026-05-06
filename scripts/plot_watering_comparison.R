library(tidyverse)
library(patchwork)

# Load data
real_df <- read_csv("scratch/real_watering_traces_2024.csv")
synth_df <- read_csv("scratch/synthetic_matching_traces_2024.csv")

# Function to clean IDs
clean_id <- function(x) {
  str_remove(as.character(x), "^0+")
}

# Reshape real data to long format
real_long <- real_df %>%
  pivot_longer(cols = -date_time, names_to = "home_id", values_to = "usage") %>%
  mutate(source = "Real", home_id = clean_id(home_id))

# Synth data
synth_long <- synth_df %>%
  rename(usage = synthetic_usage) %>%
  mutate(source = "Synthetic", home_id = clean_id(home_id))

# Combine
combined <- bind_rows(real_long, synth_long) %>%
  mutate(time_of_day = as.POSIXct(format(date_time, "%H:%M:%S"), format = "%H:%M:%S", tz = "UTC"))

# Mapping of home_ids to "Home 1", "Home 2", etc.
unique_hids <- unique(combined$home_id)
home_names <- setNames(paste("Home", seq_along(unique_hids)), unique_hids)

# Plotting Function
make_comparison_plot <- function(hid) {
  df_sub <- combined %>% filter(home_id == hid)
  display_name <- home_names[hid]
  
  ggplot(df_sub, aes(x = time_of_day, y = usage, color = source)) +
    geom_line(linewidth = 1.0, alpha = 0.8) +
    scale_color_manual(values = c("Real" = "#3498db", "Synthetic" = "#e74c3c")) +
    scale_x_datetime(date_labels = "%H:%M", date_breaks = "6 hours") +
    theme_minimal(base_size = 14) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 16, face = "bold"),
      legend.position = "none",
      axis.title = element_blank(),
      panel.background = element_rect(fill = "#fdfdfd", color = NA),
      plot.background = element_rect(fill = "#fdfdfd", color = NA)
    ) +
    labs(title = display_name)
}

# Generate 4 plots
p1 <- make_comparison_plot(unique_hids[1])
p2 <- make_comparison_plot(unique_hids[2])
p3 <- make_comparison_plot(unique_hids[3])
p4 <- make_comparison_plot(unique_hids[4])

# Combine with patchwork
final_plot <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "Real vs. Synthetic Watering Profiles (June 24, 2024)",
    subtitle = "Diffusion model predictions matching real AMI traces under identical conditions",
    caption = "Blue: Ground Truth (Box/rds) | Red: Synthetic Prediction",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 16, hjust = 0.5)
    )
  )

# Save
ggsave("images/watering_comparison_2024_ggplot.png", final_plot, width = 14, height = 10, dpi = 150)
cat("Generated 2024 4-home comparison plot with generic titles at images/watering_comparison_2024_ggplot.png\n")
