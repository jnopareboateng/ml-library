# Load libraries
library(readr)
library(tseries)
library(astsa)
library(forecast)
library(Rssa)
library(prophet)
library(ggplot2)

# Load data
crude_data <- read_csv("D:/DEV WORK/Data Science Library/ml-library/crude-oil-forecast/Modified_Data.csv")

# Explore the data
str(crude_data)
summary(crude_data)

# Visualize the data
ggplot(crude_data, aes(Date, Price)) +
    geom_line() +
    labs(
        title = "Crude Oil Prices Over Time",
        x = "Date",
        y = "Price (USD)"
    )

# Convert the data to a time series object
crude_ts <- ts(crude_data$Price, start = c(2002, 1), frequency = 12)

# Decompose the series
decomposition <- stl(crude_ts, s.window = "periodic")
plot(decomposition)

# Perform the Augmented Dickey-Fuller (ADF) test
adf_test <- adf.test(crude_ts)
print(adf_test)

# Determine the number of differences required
ndiffs(crude_ts)

# Difference the series
diff_crude_ts <- diff(crude_ts)

# Plot the differenced series
autoplot(diff_crude_ts)

# Perform ADF test on differenced series
adf_test_differenced <- adf.test(diff_crude_ts)
print(adf_test_differenced)

# Decompose the series
decomposition <- stl(diff_crude_ts, s.window = "periodic")
plot(decomposition)

# Split data into training and testing sets
train_size <- floor(0.8 * length(diff_crude_ts))
train_diff_crude_ts <- window(diff_crude_ts, start = start(diff_crude_ts), end = time(diff_crude_ts)[train_size])
test_diff_crude_ts <- window(diff_crude_ts, start = time(diff_crude_ts)[train_size + 1], end = time(diff_crude_ts)[length(diff_crude_ts)])

# ACF and PACF plots
acf2(diff_crude_ts, main = "ACF and PACF of Differenced Crude Oil Prices")

# Fit an ARIMA model
arima_model <- auto.arima(train_diff_crude_ts)
print(arima_model)

# Check the residuals
checkresiduals(arima_model)

# Forecast
forecast <- forecast(arima_model, h = length(test_diff_crude_ts))
print(forecast)

# Plot the forecast
autoplot(forecast) + autolayer(test_diff_crude_ts, series = "Test Data")

# Plot the forecast
plot(forecast, main = "Forecasted vs Actual Crude Oil Prices")
lines(test_diff_crude_ts, col = "red")
legend("topleft", legend = c("Forecasted", "Actual"), col = c("black", "red"), lty = 1)

# forecast using the best model on a quarterly basis
forecasts <- lapply(c(3, 6, 9, 12, 15, 18, 21, 24), function(h) sarima.for(crude_ts, h, 1, 1, 0))

# Singular Spectrum Analysis (SSA) and forecasting

# Perform SSA
L <- floor(length(crude_ts) / 2)  # Window length
ssa_obj <- ssa(crude_ts, L = L)

# Plot the singular values
plot(ssa_obj, type = "values")

# Reconstruct the series using the leading components (e.g., the first two components)
groups <- list(Trend = 1:2)
recon <- reconstruct(ssa_obj, groups = groups)

# Convert reconstructed series to a data frame for ggplot
recon_df <- data.frame(
    Date = as.Date(zoo::as.yearmon(time(crude_ts)), frac = 1),  # Convert to 'yearmon' and then to 'Date'
    Original = as.numeric(crude_ts),
    Reconstructed = as.numeric(recon$Trend)
)

# Plot the original and reconstructed series using ggplot2
ggplot(recon_df, aes(x = Date)) +
    geom_line(aes(y = Original, color = "Original")) +
    geom_line(aes(y = Reconstructed, color = "Reconstructed")) +
    labs(
        title = "Original and Reconstructed Crude Oil Prices",
        x = "Date",
        y = "Price (USD)"
    ) +
    scale_color_manual(values = c("Original" = "black", "Reconstructed" = "red")) +
    theme_minimal()

# SSA Forecasting
forecast_steps <- 24  # Number of steps to forecast
ssa_forecast <- rforecast(ssa_obj, groups = list(Trend = 1:2), len = forecast_steps)

# Convert SSA forecast to a data frame for ggplot
ssa_forecast_df <- data.frame(
    Date = seq(as.Date("2023-01-01"), by = "month", length.out = forecast_steps),
    Forecast = as.numeric(ssa_forecast)  # Directly use 'ssa_forecast' if it's an atomic vector
)

# Plot the SSA forecast
ggplot() +
    geom_line(data = recon_df, aes(x = Date, y = Original, color = "Original")) +
    geom_line(data = recon_df, aes(x = Date, y = Reconstructed, color = "Reconstructed")) +
    geom_line(data = ssa_forecast_df, aes(x = Date, y = Forecast, color = "Forecast")) +
    labs(
        title = "SSA Forecast vs Actual Crude Oil Prices",
        x = "Date",
        y = "Price (USD)"
    ) +
    scale_color_manual(values = c("Original" = "black", "Reconstructed" = "red", "Forecast" = "blue")) +
    theme_minimal() +
    guides(color = guide_legend(title = "Series"))


# Evaluate the forecast accuracy
forecast_accuracy <- accuracy(ssa_forecast, crude_ts)

# Print the forecast accuracy
print(forecast_accuracy)