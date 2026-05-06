library(tidyverse)
library(patchwork)

CKPT <- "/Users/carson/Library/CloudStorage/Box-Box/models copy/water_diffusion_20260504_213400-step=187500.ckpt"

seasons <- list(
  summer = list(
    real   = "scratch/real_watering_traces_2024.csv",
    synth  = "scratch/synthetic_matching_traces_2024.csv",
    label  = "Summer (June 24, 2024)"
  ),
  winter = list(
    real   = "scratch/real_traces_winter_2024.csv",
    synth  = "scratch/synthetic_traces_winter_2024.csv",
    label  = "Winter (January 30, 2024)"
  ),
  spring = list(
    real   = "scratch/real_traces_spring_2024.csv",
    synth  = "scratch/synthetic_traces_spring_2024.csv",
    label  = "Spring (May 25, 2024)"
  ),
  fall   = list(
    real   = "scratch/real_traces_fall_2024.csv",
    synth  = "scratch/synthetic_traces_fall_2024.csv",
    label  = "Fall (September 26, 2024)"
  )
)

clean_id <- function(x) str_remove(as.character(x), "^0+")

make_season_plot <- function(season_name, info) {
  real_df  <- read_csv(info$real,  show_col_types = FALSE)
  synth_df <- read_csv(info$synth, show_col_types = FALSE)

  real_long <- real_df %>%
    pivot_longer(-date_time, names_to = "home_id", values_to = "usage") %>%
    mutate(source = "Real", home_id = clean_id(home_id))

  synth_long <- synth_df %>%
    rename(usage = synthetic_usage) %>%
    mutate(source = "Synthetic", home_id = clean_id(home_id))

  combined <- bind_rows(real_long, synth_long) %>%
    mutate(time_of_day = as.POSIXct(format(date_time, "%H:%M:%S"),
                                     format = "%H:%M:%S", tz = "UTC"))

  unique_hids <- unique(combined$home_id)
  home_names  <- setNames(paste("Home", seq_along(unique_hids)), unique_hids)

  make_panel <- function(hid) {
    df_sub <- combined %>% filter(home_id == hid)
    ggplot(df_sub, aes(x = time_of_day, y = usage, color = source)) +
      geom_line(linewidth = 1.0, alpha = 0.85) +
      scale_color_manual(values = c("Real" = "#3498db", "Synthetic" = "#e74c3c")) +
      scale_x_datetime(date_labels = "%H:%M", date_breaks = "6 hours") +
      theme_minimal(base_size = 14) +
      theme(
        panel.grid.minor  = element_blank(),
        plot.title        = element_text(size = 15, face = "bold"),
        legend.position   = "bottom",
        axis.title        = element_blank(),
        panel.background  = element_rect(fill = "#fdfdfd", color = NA),
        plot.background   = element_rect(fill = "#fdfdfd", color = NA)
      ) +
      labs(title = home_names[hid], color = "Data Source")
  }

  panels <- map(unique_hids, make_panel)

  combined_plot <- (panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]]) +
    plot_layout(guides = "collect") +
    plot_annotation(
      title    = paste("Real vs. Synthetic —", info$label),
      subtitle = "Diffusion model conditioned on actual climate & behavioral features",
      caption  = "Blue: Ground Truth AMI  |  Red: Synthetic (Diffusion Model)",
      theme    = theme(
        plot.title    = element_text(size = 22, face = "bold",  hjust = 0.5),
        plot.subtitle = element_text(size = 15, hjust = 0.5),
        plot.caption  = element_text(size = 11)
      )
    )
  
  combined_plot & theme(legend.position = "bottom")
}

# Generate and save each season
for (season_name in names(seasons)) {
  info <- seasons[[season_name]]
  if (!file.exists(info$synth)) {
    cat("Skipping", season_name, "— synthetic file not yet generated.\n")
    next
  }
  p <- make_season_plot(season_name, info)
  outfile <- paste0("images/comparison_", season_name, "_2024_ggplot.png")
  ggsave(outfile, p, width = 14, height = 10, dpi = 150)
  cat("Saved", outfile, "\n")
}
