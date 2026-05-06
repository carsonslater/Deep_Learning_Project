library(tidyverse)
library(viridis)

# Load synthetic samples
samples_df <- read_csv("scratch/synthetic_samples_raw.csv")

# Create a heatmap of the 50 synthetic traces
p_heatmap <- ggplot(samples_df, aes(x = interval, y = factor(sample_id), fill = usage)) +
  geom_tile() +
  scale_fill_viridis_c(option = "magma", name = "Gallons") +
  scale_x_continuous(expand = c(0, 0), breaks = seq(0, 96, 12)) +
  labs(
    title = "Stochastic Trace Generation (Diffusion POC)",
    subtitle = "Heatmap of 50 synthetic 24-hour profiles",
    x = "15-minute Interval",
    y = "Sample ID"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    axis.text.y = element_blank(),
    plot.title = element_text(face = "bold", size = 18),
    legend.position = "right"
  )

ggsave("images/synthetic_samples_heatmap_magma.png", p_heatmap, width = 12, height = 6, dpi = 150)
cat("Saved images/synthetic_samples_heatmap_magma.png\n")

# Also remake the line plot but with better ggplot styling
mean_profile <- samples_df %>%
  group_by(interval) %>%
  summarize(usage = mean(usage), .groups = "drop")

p_lines <- ggplot(samples_df, aes(x = interval, y = usage, group = sample_id)) +
  geom_line(alpha = 0.1, color = "#2c3e50") +
  geom_line(data = mean_profile, aes(x = interval, y = usage, group = 1), 
            color = "#e74c3c", linewidth = 1.2) +
  labs(
    title = "Synthetic Usage Envelopes",
    subtitle = "50 generated samples (black) vs. Population Mean (red)",
    x = "Interval",
    y = "Gallons"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))

ggsave("images/synthetic_samples_poc_ggplot.png", p_lines, width = 12, height = 6, dpi = 150)
cat("Saved images/synthetic_samples_poc_ggplot.png\n")
