library(tidyverse)
library(viridis)
library(patchwork)

# Load the data
dist_df <- read_csv("scratch/distribution_data.csv")
profile_df <- read_csv("scratch/profile_data.csv")

# 1. Profile Comparison (Line plot)
profile_long <- profile_df %>%
  pivot_longer(cols = c(real_mean, synthetic_mean), names_to = "source", values_to = "usage") %>%
  mutate(source = if_else(source == "real_mean", "Real", "Synthetic"))

p1 <- ggplot(profile_long, aes(x = interval, y = usage, color = source)) +
  geom_line(linewidth = 1.5, alpha = 0.9) +
  scale_color_manual(values = c("Real" = "#3498db", "Synthetic" = "#e74c3c")) +
  labs(title = "Mean Diurnal Profile", subtitle = "Aggregate alignment across 96 intervals", x = "Interval", y = "Gallons") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

# 2. Magnitude Distribution (The Magma Heat-density plot)
# We can use geom_density_2d_filled or just a stylized density
p2 <- ggplot(dist_df, aes(x = usage, fill = source)) +
  geom_density(alpha = 0.7, color = "white") +
  scale_x_log10(breaks = c(0.1, 1, 10, 100), labels = c("0.1", "1", "10", "100")) +
  # Using magma for the fill based on a dummy variable or just styling
  scale_fill_viridis_d(option = "magma", begin = 0.3, end = 0.8) +
  labs(title = "Magnitude Distribution", subtitle = "Log-scale density comparison", x = "Usage (Gallons)", y = "Density") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

# Combine
combined <- p1 + p2 + 
  plot_annotation(
    title = "Model Validation: Distributional Fidelity",
    subtitle = "Comparing Real AMI population vs. Synthetic Diffusion samples",
    theme = theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
                  plot.subtitle = element_text(size = 14, hjust = 0.5))
  )

ggsave("images/distributional_validation_magma.png", combined, width = 14, height = 7, dpi = 150)
cat("Saved images/distributional_validation_magma.png\n")
