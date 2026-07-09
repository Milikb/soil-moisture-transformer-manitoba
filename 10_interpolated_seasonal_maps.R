# Seasonal interpolated observed and model-predicted soil moisture maps from station-level results.

library(ggplot2)
library(sf)
library(dplyr)
library(readr)
library(gstat)
library(raster)
library(sp)
library(cowplot)
library(ggspatial)
source("R/plotting_helpers.R")

# ---- User settings ----
results_dir <- "outputs/model_results/depth_GW/Transformer_20cm_withGW"
mb_prov_shp <- "data/shapefiles/Manitoba_Provincial_Boundary.shp"
mb_lakes_shp <- "data/shapefiles/lakes_in_manitoba.shp"
output_dir <- "outputs/figures/interpolated_maps"
output_file <- file.path(output_dir, "seasonal_interpolated_observed_modeled_20cm.png")
create_dir(output_dir)

seasonal_dates <- as.Date(c("2019-07-15"))
map_lat_max <- 51.5
color_limits <- c(0, 50)
grid_resolution <- 100

# ---- Spatial layers ----
manitoba <- st_read(mb_prov_shp, quiet = TRUE) |> st_transform(4326) |> st_make_valid()
lakes_mb <- st_read(mb_lakes_shp, quiet = TRUE) |> st_transform(4326) |> st_make_valid()

base_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    plot.margin = margin(2, 0.5, 2, 0.5)
  )

read_station_data_for_date <- function(results_dir, input_date) {
  files <- list.files(results_dir, pattern = "\\.csv$", full.names = TRUE)
  out <- lapply(files, function(file) {
    df <- read_csv(file, col_types = cols()) |>
      mutate(Date = as.Date(Date)) |>
      filter(Date == input_date)
    if (nrow(df) == 0) return(NULL)
    df$Station <- tools::file_path_sans_ext(basename(file))
    df
  })
  bind_rows(out)
}

interpolate_variable <- function(soil_data, value_col, grid_resolution = 100) {
  sp_data <- soil_data
  coordinates(sp_data) <- ~ LongDD + LatDD
  proj4string(sp_data) <- CRS("+proj=longlat +datum=WGS84")

  grd <- expand.grid(
    LongDD = seq(min(soil_data$LongDD) - 0.1, max(soil_data$LongDD) + 0.1, length.out = grid_resolution),
    LatDD = seq(min(soil_data$LatDD) - 0.1, max(soil_data$LatDD) + 0.1, length.out = grid_resolution)
  )
  coordinates(grd) <- ~ LongDD + LatDD
  gridded(grd) <- TRUE
  proj4string(grd) <- CRS("+proj=longlat +datum=WGS84")

  f <- as.formula(paste(value_col, "~ 1"))
  r <- raster(predict(gstat(formula = f, data = sp_data, nmax = 30), grd))
  as.data.frame(raster::as.data.frame(r, xy = TRUE)) |>
    setNames(c("LongDD", "LatDD", value_col)) |>
    mutate(across(everything(), as.numeric))
}

plot_interpolated_map <- function(map_df, value_col, title, fill_name, show_y = TRUE, show_x = TRUE, show_legend = TRUE) {
  map_df <- map_df |> filter(LatDD < map_lat_max)
  manitoba_crop <- st_crop(
    manitoba,
    xmin = min(map_df$LongDD, na.rm = TRUE), xmax = max(map_df$LongDD, na.rm = TRUE),
    ymin = min(map_df$LatDD, na.rm = TRUE), ymax = max(map_df$LatDD, na.rm = TRUE)
  )

  ggplot() +
    geom_raster(data = map_df, aes(x = LongDD, y = LatDD, fill = .data[[value_col]])) +
    geom_sf(data = lakes_mb, fill = "lightblue", color = NA) +
    geom_sf(data = manitoba_crop, fill = NA, color = "black", linewidth = 0.5) +
    scale_fill_gradientn(colours = soil_moisture_palette, limits = color_limits, name = fill_name, guide = if (show_legend) "colourbar" else "none") +
    labs(title = title, x = "Longitude", y = "Latitude") +
    annotation_scale(location = "bl", width_hint = 0.25, bar_cols = c("grey60", "white")) +
    annotation_north_arrow(location = if (show_y) "tl" else "tr", which_north = "true", style = north_arrow_fancy_orienteering()) +
    coord_sf(xlim = range(map_df$LongDD, na.rm = TRUE), ylim = range(map_df$LatDD, na.rm = TRUE), expand = FALSE) +
    base_theme +
    theme(
      axis.title.y = if (show_y) element_text(size = 12) else element_blank(),
      axis.text.y = if (show_y) element_text(size = 10) else element_blank(),
      axis.ticks.y = if (show_y) element_line() else element_blank(),
      axis.title.x = if (show_x) element_text(size = 12) else element_blank(),
      axis.text.x = if (show_x) element_text(size = 10) else element_blank(),
      axis.ticks.x = if (show_x) element_line() else element_blank()
    )
}

# ---- Build maps ----
all_plots <- list()
for (input_date in seasonal_dates) {
  soil_data <- read_station_data_for_date(results_dir, input_date)
  if (nrow(soil_data) == 0) next

  obs_df <- interpolate_variable(soil_data, "Observed_Test", grid_resolution) |>
    rename(Observed_Soil = Observed_Test)
  pred_df <- interpolate_variable(soil_data, "Predicted_Test", grid_resolution) |>
    rename(Modeled_Soil = Predicted_Test)

  all_plots <- c(
    all_plots,
    list(
      plot_interpolated_map(obs_df, "Observed_Soil", paste("Observed –", format(input_date, "%b %d, %Y")), "Observed SM (%)", show_y = TRUE, show_x = TRUE, show_legend = FALSE),
      plot_interpolated_map(pred_df, "Modeled_Soil", paste("Modeled –", format(input_date, "%b %d, %Y")), "Soil moisture (%)", show_y = FALSE, show_x = TRUE, show_legend = TRUE)
    )
  )
}

final_plot <- plot_grid(plotlist = all_plots, ncol = 2, align = "hv", rel_widths = c(1, 1))
ggsave(output_file, final_plot, width = 18, height = 8, dpi = 600, units = "in", bg = "white")
