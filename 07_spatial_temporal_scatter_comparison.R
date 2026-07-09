# Comparison of spatial and temporal Transformer performance across temporal resolutions.

library(tidyverse)
library(tools)
library(ggpointdensity)
library(cowplot)
library(scales)
library(ggExtra)
library(viridisLite)
source("R/plotting_helpers.R")

# ---- User settings ----
spatial_dirs <- list(
  Hourly = "outputs/model_results/spatial/hourly_5cm",
  Daily = "outputs/model_results/spatial/daily_5cm",
  Weekly = "outputs/model_results/spatial/weekly_5cm"
)

temporal_dirs <- list(
  Hourly = "outputs/model_results/temporal/hourly_5cm",
  Daily = "outputs/model_results/temporal/daily_5cm_withGW",
  Weekly = "outputs/model_results/temporal/weekly_5cm"
)

output_dir <- "outputs/figures/spatial_temporal"
output_file <- file.path(output_dir, "spatial_temporal_comparison.png")
create_dir(output_dir)

# ---- Read data and compute metrics ----
spatial_data <- lapply(spatial_dirs, read_spatial_results)
temporal_data <- lapply(temporal_dirs, read_temporal_results)

spatial_stats <- lapply(spatial_data, model_stats)
temporal_stats <- lapply(temporal_data, model_stats)

# ---- Create panels ----
spatial_plots <- Map(
  function(df, st, nm, show_y) plot_density_scatter(df, st, row_label = "Spatial", show_x = FALSE, show_y = show_y),
  spatial_data, spatial_stats, names(spatial_data), c(TRUE, FALSE, FALSE)
)

temporal_plots <- Map(
  function(df, st, nm, show_y) plot_density_scatter(df, st, row_label = "Temporal", show_x = TRUE, show_y = show_y),
  temporal_data, temporal_stats, names(temporal_data), c(TRUE, FALSE, FALSE)
)

top_row <- cowplot::plot_grid(plotlist = spatial_plots, ncol = 3, labels = NULL, align = "hv", axis = "tblr")
bottom_row <- cowplot::plot_grid(plotlist = temporal_plots, ncol = 3, labels = NULL, align = "hv", axis = "tblr")

final_plot <- cowplot::plot_grid(top_row, bottom_row, ncol = 1, labels = NULL, align = "v", axis = "lr")
final_plot <- cowplot::ggdraw(final_plot) + theme(plot.background = element_rect(fill = "white", color = NA))

ggsave(output_file, final_plot, dpi = 600, width = 11.5, height = 7.5, units = "in", bg = "white")
