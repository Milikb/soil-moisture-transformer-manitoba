# Soil moisture data availability by station.

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(lubridate)
source("R/plotting_helpers.R")

# ---- User settings ----
soil_data_dir <- "data/agriculture_hourly"
coord_csv <- "data/lat.csv"
output_dir <- "outputs/figures/data_availability"
output_file <- file.path(output_dir, "soil_moisture_data_availability.png")
create_dir(output_dir)

soil_moisture_col <- "Soil_TP5_VMC"
date_start <- as.Date("2018-01-01")
date_end <- as.Date("2025-12-31")

# ---- Load data ----
soil <- read_csv(list.files(soil_data_dir, pattern = "\\.csv$", full.names = TRUE), show_col_types = FALSE) |>
  transmute(
    Station = tolower(Station),
    Date = as.Date(as.POSIXct(TMSTAMP, tz = "UTC")),
    SM = as.numeric(.data[[soil_moisture_col]])
  ) |>
  filter(Date >= date_start, Date <= date_end)

coords <- read_csv(coord_csv, show_col_types = FALSE) |>
  transmute(Station = tolower(StationName), LatDD = as.numeric(LatDD)) |>
  arrange(desc(LatDD))

station_names <- soil |>
  filter(!is.na(SM)) |>
  count(Station) |>
  filter(n > 0) |>
  pull(Station)

full_grid <- expand.grid(
  Date = seq(date_start, date_end, by = "day"),
  Station = station_names
)

availability <- full_grid |>
  left_join(soil |> group_by(Date, Station) |> summarise(SM = mean(SM, na.rm = TRUE), .groups = "drop"), by = c("Date", "Station")) |>
  inner_join(coords, by = "Station") |>
  mutate(
    Station_factor = factor(Station, levels = coords$Station[coords$Station %in% station_names]),
    Status = if_else(is.na(SM), "Missing", "Available")
  )

p <- ggplot(availability, aes(Date, Station_factor)) +
  geom_point(aes(color = Status), size = 0.7, alpha = 0.7) +
  scale_x_date(
    breaks = seq(date_start, date_end, by = "1 year"),
    labels = format(seq(date_start, date_end, by = "1 year"), "%Y")
  ) +
  scale_color_manual(values = c("Available" = "steelblue", "Missing" = "grey85"), name = "Data status") +
  labs(x = "Date", y = "Station") +
  theme_minimal(base_size = 22) +
  theme(
    axis.text.y = element_text(size = 20, hjust = 0),
    axis.text.x = element_text(size = 20, hjust = 0),
    axis.title.x = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 20, face = "bold"),
    legend.position = "bottom"
  )

ggsave(output_file, p, width = 20, height = 18, dpi = 600, bg = "white")
