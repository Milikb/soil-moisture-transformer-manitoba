# Station-level NSE point map.

library(ggplot2)
library(sf)
library(dplyr)
library(readr)
library(ggspatial)
source("R/plotting_helpers.R")

# ---- User settings ----
results_dir <- "outputs/model_results/depth_GW/Transformer_20cm_withGW"
coord_csv <- "data/lat.csv"
mb_prov_shp <- "data/shapefiles/Manitoba_Provincial_Boundary.shp"
mb_lakes_shp <- "data/shapefiles/lakes_in_manitoba.shp"
output_dir <- "outputs/figures/nse_maps"
output_file <- file.path(output_dir, "station_nse_map_20cm.png")
create_dir(output_dir)

# ---- Compute station NSE ----
files <- list.files(results_dir, pattern = "\\.csv$", full.names = TRUE)
station_stats <- lapply(files, function(file) {
  df <- read_csv(file, show_col_types = FALSE) |> tidyr::drop_na()
  model_stats(df) |> mutate(Station = tolower(extract_station_name(file)))
}) |>
  bind_rows()

coords <- read_csv(coord_csv, show_col_types = FALSE) |>
  transmute(Station = tolower(StationName), LongDD = as.numeric(LongDD), LatDD = as.numeric(LatDD))

nse_points <- station_stats |>
  left_join(coords, by = "Station") |>
  filter(is.finite(NSE), !is.na(LongDD), !is.na(LatDD))

map_sf <- st_as_sf(nse_points, coords = c("LongDD", "LatDD"), crs = 4326)

# ---- Spatial layers ----
mb_prov <- st_read(mb_prov_shp, quiet = TRUE) |> st_transform(4326) |> st_make_valid()
mb_lakes <- st_read(mb_lakes_shp, quiet = TRUE) |> st_transform(4326) |> st_make_valid()

# ---- Plot ----
p <- ggplot() +
  geom_sf(data = mb_prov, fill = "gray95", color = "gray85") +
  geom_sf(data = mb_lakes, fill = "#CDE6FF", color = NA, alpha = 0.6) +
  geom_sf(data = map_sf, aes(fill = NSE), shape = 21, color = "white", stroke = 0.6, size = 3.2, alpha = 0.95) +
  scale_fill_gradient2(
    high = "darkgreen", mid = "yellow", low = "red",
    midpoint = 0.5, limits = c(0, 1.1), oob = scales::squish, name = "NSE"
  ) +
  coord_sf(xlim = c(-102, -94), ylim = c(48.5, 51.5), expand = FALSE, datum = NA) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = c(0.85, 0.66),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3),
    legend.key.height = unit(4.5, "mm"),
    legend.key.width = unit(7, "mm"),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(output_file, p, dpi = 600, width = 180, height = 135, units = "mm", bg = "white")
