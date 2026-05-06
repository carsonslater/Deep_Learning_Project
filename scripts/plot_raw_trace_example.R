library(tidyverse)
library(lubridate)

# Load data
df <- read_csv("scratch/real_traces_summer_2024.csv") %>%
  select(date_time, usage = `8242050119`) %>%
  mutate(date_time = as_datetime(date_time))

# Plot
p <- ggplot(df, aes(x = date_time, y = usage)) +
  geom_area(fill = "#003024", alpha = 0.6) +
  geom_line(color = "#003024", linewidth = 1) +
  scale_x_datetime(date_labels = "%H:%M", date_breaks = "4 hours") +
  labs(
    title = "Raw AMI Flow Trace Example",
    subtitle = "Single-family residential meter (15-minute resolution)",
    x = "Time of Day (June 24, 2024)",
    y = "Water Usage (Gallons)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave("images/raw_trace_example.png", p, width = 12, height = 6, dpi = 150)
cat("Saved images/raw_trace_example.png\n")
