# Average observed and predicted soil moisture time series across stations.

library(ggplot2)
library(dplyr)
library(readr)
library(scales)
source("R/plotting_helpers.R")

# ---- User settings ----
results_dir <- "outputs/model_results/depth_GW/Transformer_20cm_withGW"
output_dir <- "outputs/figures/average_timeseries"
output_file <- file.path(output_dir, "average_observed_predicted_soil_moisture.png")
create_dir(output_dir)

depth_cm <- 20

# ---- Load station results and average by date ----
files <- list.files(results_dir, pattern = "\\.csv$", full.names = TRUE)
soil_data <- lapply(files, function(file) {
  read_csv(file, show_col_types = FALSE) |>
    mutate(Station = extract_station_name(file), Date = as.POSIXct(Date, tz = "UTC"))
}) |>
  bind_rows()

mean_series <- soil_data |>
  group_by(Date) |>
  summarise(
    Observed = mean(Observed_Test, na.rm = TRUE),
    Predicted = mean(Predicted_Test, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(Date)

# ---- Metrics and test shading ----
test_start_idx <- ceiling(0.8 * nrow(mean_series))
test_start_date <- mean_series$Date[test_start_idx]
test_end_date <- max(mean_series$Date, na.rm = TRUE)
metrics <- model_stats(mean_series, observed_col = "Observed", predicted_col = "Predicted")
label_txt <- sprintf("NSE= %.2f, Cor= %.2f, RMSE= %.2f", metrics$NSE, metrics$cor, metrics$RMSE)

# ---- Plot ----
p <- ggplot(mean_series, aes(Date)) +
  geom_rect(aes(xmin = test_start_date, xmax = test_end_date, ymin = -Inf, ymax = Inf, fill = "Test period"), inherit.aes = FALSE, alpha = 0.4) +
  geom_line(aes(y = Observed, colour = "Observed"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, colour = "Predicted"), linewidth = 0.8, linetype = "dashed", alpha = 0.7) +
  scale_color_manual(name = NULL, values = c("Observed" = "black", "Predicted" = "#1f77b4")) +
  scale_fill_manual(name = NULL, values = c("Test period" = "grey80")) +
  scale_x_datetime(breaks = breaks_width("1 year"), labels = label_date("%Y")) +
  scale_y_continuous(limits = c(8, 45), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Time", y = paste0("Soil moisture (%) ", depth_cm, " cm")) +
  annotate("label", label = label_txt, x = min(mean_series$Date, na.rm = TRUE), y = 40, hjust = 0, size = 4) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.2, color = "grey80"),
    panel.border = element_rect(color = "black", linewidth = 0.3, fill = NA),
    legend.position = c(0.8, 0.07),
    legend.direction = "horizontal",
    legend.box.background = element_rect(color = "black", linewidth = 0.5, fill = NA)
  )

ggsave(output_file, p, width = 9, height = 5, dpi = 500, bg = "white")
