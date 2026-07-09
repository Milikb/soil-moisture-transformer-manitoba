# Compare station-interpolated observed soil moisture and raster-based model soil moisture for selected dates.

library(terra)
library(sf)
library(ggplot2)
library(ggspatial)
library(dplyr)
library(scales)
library(sp)
library(raster)
library(gstat)
library(cowplot)
library(readr)
source("R/plotting_helpers.R")

# ---- User settings ----
station_results_dir <- "outputs/model_results/depth_GW/Transformer_20cm_withGW"
raster_dir <- "outputs/raster_predictions/predicted_soil_moisture_20cm"
raster_template <- "soil_moisture_20cm_%Y%m%d_UTM14.tif"
mb_prov_shp <- "data/shapefiles/Manitoba_Provincial_Boundary.shp"
mb_lakes_shp <- "data/shapefiles/lakes_in_manitoba.shp"
output_dir <- "outputs/figures/raster_interpolated_maps"
create_dir(output_dir)

start_date <- as.Date("2024-03-14")
end_date <- as.Date("2024-03-14")
map_lat_max <- 51.5
color_limits <- c(0, 50)

manitoba <- st_read(mb_prov_shp, quiet = TRUE) |> st_transform(4326) |> st_make_valid()
lakes_mb <- st_read(mb_lakes_shp, quiet = TRUE) |> st_transform(4326) |> st_make_valid()

base_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 12)
  )

read_station_data_for_date <- function(input_date) {
  files <- list.files(station_results_dir, pattern = "\\.csv$", full.names = TRUE)
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

interpolate_observed <- function(soil_data) {
  sp_data <- soil_data
  coordinates(sp_data) <- ~ LongDD + LatDD
  proj4string(sp_data) <- CRS("+proj=longlat +datum=WGS84")

  grd <- expand.grid(
    LongDD = seq(min(soil_data$LongDD) - 0.1, max(soil_data$LongDD) + 0.1, length.out = 100),
    LatDD = seq(min(soil_data$LatDD) - 0.1, max(soil_data$LatDD) + 0.1, length.out = 100)
  )
  coordinates(grd) <- ~ LongDD + LatDD
  gridded(grd) <- TRUE
  proj4string(grd) <- CRS("+proj=longlat +datum=WGS84")

  observed_raster <- raster(predict(gstat::gstat(formula = Observed_Test ~ 1, data = sp_data, nmax = 30), grd))
  as.data.frame(raster::as.data.frame(observed_raster, xy = TRUE)) |>
    setNames(c("LongDD", "LatDD", "Observed_Soil")) |>
    filter(LatDD < map_lat_max)
}

load_raster_map <- function(raster_path, crop_boundary) {
  sm <- rast(raster_path)
  names(sm) <- "Modeled_Soil"
  sm[sm == -9999] <- NA
  sm[sm < 0] <- NA

  crop_boundary_utm <- st_transform(crop_boundary, crs(sm))
  sm_clip <- crop(sm, vect(crop_boundary_utm)) |>
    mask(vect(crop_boundary_utm))
  sm_ll <- project(sm_clip, "EPSG:4326", method = "bilinear")

  as.data.frame(sm_ll, xy = TRUE, na.rm = TRUE) |>
    setNames(c("LongDD", "LatDD", "Modeled_Soil")) |>
    filter(LatDD < map_lat_max)
}

plot_map <- function(df, value_col, title, show_y = TRUE) {
  manitoba_crop <- st_crop(
    manitoba,
    xmin = min(df$LongDD, na.rm = TRUE), xmax = max(df$LongDD, na.rm = TRUE),
    ymin = min(df$LatDD, na.rm = TRUE), ymax = max(df$LatDD, na.rm = TRUE)
  )

  ggplot() +
    geom_raster(data = df, aes(x = LongDD, y = LatDD, fill = .data[[value_col]]), interpolate = TRUE) +
    geom_sf(data = lakes_mb, fill = "lightblue", color = NA) +
    geom_sf(data = manitoba_crop, fill = NA, color = "black", linewidth = 0.5) +
    scale_fill_gradientn(colours = soil_moisture_palette, name = "SM (%)", limits = color_limits, oob = squish) +
    labs(title = title, x = "Longitude", y = "Latitude") +
    annotation_scale(location = "bl", width_hint = 0.25, bar_cols = c("grey60", "white")) +
    annotation_north_arrow(location = "tl", which_north = "true", style = north_arrow_fancy_orienteering()) +
    coord_sf(xlim = range(df$LongDD, na.rm = TRUE), ylim = range(df$LatDD, na.rm = TRUE), expand = FALSE) +
    base_theme +
    theme(
      axis.title.y = if (show_y) element_text(size = 16) else element_blank(),
      axis.text.y = if (show_y) element_text(size = 12) else element_blank(),
      axis.ticks.y = if (show_y) element_line() else element_blank()
    )
}

# ---- Generate maps ----
for (input_date in seq(start_date, end_date, by = "day")) {
  station_data <- read_station_data_for_date(input_date)
  if (nrow(station_data) == 0) next

  obs_df <- interpolate_observed(station_data)
  manitoba_crop <- st_crop(
    manitoba,
    xmin = min(obs_df$LongDD, na.rm = TRUE), xmax = max(obs_df$LongDD, na.rm = TRUE),
    ymin = min(obs_df$LatDD, na.rm = TRUE), ymax = max(obs_df$LatDD, na.rm = TRUE)
  )

  raster_path <- file.path(raster_dir, format(input_date, raster_template))
  if (!file.exists(raster_path)) next
  mod_df <- load_raster_map(raster_path, manitoba_crop)

  observed_plot <- plot_map(obs_df, "Observed_Soil", paste("Observed –", format(input_date, "%b %d, %Y")), show_y = TRUE)
  modeled_plot <- plot_map(mod_df, "Modeled_Soil", paste("Modeled –", format(input_date, "%b %d, %Y")), show_y = FALSE)

  side_by_side <- plot_grid(observed_plot, modeled_plot, ncol = 2, align = "hv", labels = c("A", "B"), label_size = 16)
  ggsave(file.path(output_dir, paste0("observed_raster_comparison_", input_date, ".png")), side_by_side, width = 18, height = 10, dpi = 300, units = "in", bg = "white")
}
