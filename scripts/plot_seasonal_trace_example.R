library(tidyverse)
library(lubridate)

# Helper to load and tag
load_season <- function(file, season_name) {
  read_csv(file) %>%
    select(date_time, usage = `8242050119`) %>%
    mutate(
      date_time = as_datetime(date_time),
      # Normalize time to a single day for plotting on same X axis
      time_only = update(date_time, year = 2000, month = 1, day = 1),
      season = factor(season_name, levels = c("Winter", "Spring", "Summer", "Fall"))
    )
}

df_all <- bind_rows(
  load_season("scratch/real_traces_winter_2024.csv", "Winter"),
  load_season("scratch/real_traces_spring_2024.csv", "Spring"),
  load_season("scratch/real_traces_summer_2024.csv", "Summer"),
  load_season("scratch/real_traces_fall_2024.csv", "Fall")
)

# Plot with facets
p <- ggplot(df_all, aes(x = time_only, y = usage)) +
  geom_line(color = "#003024", linewidth = 1.2) +
  facet_wrap(~season, scales = "fixed", ncol = 2) +
  scale_x_datetime(date_labels = "%H:%M", date_breaks = "6 hours") +
  labs(
    title = "Seasonal Water Usage Profile: Residential Home",
    subtitle = "Comparing typical daily flow traces across 2024 seasons (Same Meter)",
    x = "Time of Day",
    y = "Water Usage (Gallons)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#eeeeee", fill = NA)
  )

ggsave("images/seasonal_trace_example.png", p, width = 12, height = 8, dpi = 150)
cat("Saved images/seasonal_trace_example.png\n")
