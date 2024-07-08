# Load necessary packages
library(Rssa)
library(forecast)
library(stats)
library(cluster)

# Load data
load_data <- function() {
  data_raw <- read.csv(file.choose(), sep = ",", header = TRUE)
  return(data_raw)
}

# Preprocess data
preprocess_data <- function(data_raw) {
  data_ts <- ts(data_raw, start = c(2000, 1), frequency = 12)
  return(data_ts)
}

# Perform SSA
perform_ssa <- function(data_ts, L) {
  ssa_result <- ssa(data_ts, L = L, kind = "1d-ssa")
  return(ssa_result)
}

# Reconstruct components
reconstruct_components <- function(ssa_result) {
  # Extract eigenvalues and eigenvectors
  eigen_decomp <- ssa_result$U
  
  # Reconstruct components using hierarchical clustering
  component_groups <- hclust(dist(eigen_decomp), method = "ward.D2")
  component_labels <- cutree(component_groups, k = 3)
  
  # Reconstruct components
  trend_component <- reconstruct(ssa_result, groups = list(which(component_labels == 1)))
  seasonal_component <- reconstruct(ssa_result, groups = list(which(component_labels == 2)))
  residual_component <- reconstruct(ssa_result, groups = list(which(component_labels == 3)))
  
  return(list(trend_component = trend_component, seasonal_component = seasonal_component, residual_component = residual_component))
}

# Forecast
forecast_components <- function(trend_component, seasonal_component, residual_component) {
  # Forecast using ARIMA
  forecast_trend <- forecast(auto.arima(trend_component), h = 12)
  forecast_seasonal <- forecast(auto.arima(seasonal_component), h = 12)
  forecast_residual <- forecast(auto.arima(residual_component), h = 12)
  
  # Combine forecasts (using appropriate combination method)
  forecast_combined <- forecast_trend$mean + forecast_seasonal$mean + forecast_residual$mean
  
  return(forecast_combined)
}

# Plot results
plot_results <- function(data_ts, trend_component, seasonal_component, residual_component, forecast_combined) {
  # Plot original data and forecast
  plot.ts(cbind(data_ts, forecast_combined), main = "Original Data and Forecast", col = c("blue", "red"))
  
  # Plot components
  plot.ts(cbind(trend_component, seasonal_component, residual_component), main = "Components", col = c("green", "orange", "purple"))
}

# Main script
data_raw <- load_data()
data_ts <- preprocess_data(data_raw)

# Set window length (L)
L <- 12 # or use a heuristic to set L

ssa_result <- perform_ssa(data_ts, L)

component_list <- reconstruct_components(ssa_result)
trend_component <- component_list$trend_component
seasonal_component <- component_list$seasonal_component
residual_component <- component_list$residual_component

forecast_combined <- forecast_components(trend_component, seasonal_component, residual_component)

plot_results(data_ts, trend_component, seasonal_component, residual_component, forecast_combined)



