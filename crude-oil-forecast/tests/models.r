# Load required packages
library(tseries)
library(astsa)
library(forecast)
library(Rssa)
library(prophet)

# Load crude oil prices data
crude <- read.csv("D:/DEV WORK/Data Science Library/ml-library/crude-oil-forecast/Modified_Data.csv")
# Create time series object
crude_vector <- as.vector(crude$Price)
Crude <- ts(crude_vector, start = 2002, frequency = 12)

# Plot the crude oil prices time series
plot(Crude, main = "Crude Oil Prices over 20 years")

# Perform ADF test for stationarity
adf.test(Crude)

# Determine the number of differences required
ndiffs(Crude) # Trend differencing
nsdiffs(Crude) # Seasonal differencing

# Difference the series
dCrude <- diff(Crude)

# Perform ADF test on differenced series
adf.test(dCrude)

# Plot the differenced series
autoplot(dCrude)

# Build and summarize SARIMA models
BestModel <- auto.arima(Crude, seasonal = FALSE)
summary(BestModel)

# Fit several SARIMA models and compare
fits <- list(
    sarima(Crude, 1, 1, 0),
    sarima(Crude, 1, 1, 2),
    sarima(Crude, 1, 1, 6),
    sarima(Crude, 5, 1, 1),
    sarima(Crude, 5, 1, 2),
    sarima(Crude, 5, 1, 6)
)

# Forecast using the best model on a quarterly basis
forecasts <- lapply(c(3, 6, 9, 12), function(h) sarima.for(Crude, h, 1, 1, 0))

# Singular Spectrum Analysis (SSA) and forecasting
ssa_data <- read.csv("D:/DEV WORK/Data Science Library/ml-library/crude-oil-forecast/ssa_crude.csv", sep = ",", header = TRUE)
ssa_data.ts <- ts(ssa_data, start = c(2000, 1), frequency = 12)
plot(ssa_data.ts)

n <- length(ssa_data.ts)
p <- 20
insample <- ssa_data.ts[1:(n - p)]
outsample <- ssa_data.ts[(n - p + 1):n]

L <- n / 2 # Trial and error (L ≤ N/2)
K <- n - L + 1

# Perform SSA
s <- ssa(insample, L = L, kind = "1d-ssa")
plot(s)

# Grouping and reconstructing components
r <- reconstruct(s, groups = list(Trend = c(1), seasonal = c(2, 3), season2 = c(4)), len = p)
plot(wcor(s, groups = list(Trend = c(1), seasonal = c(2, 3), season2 = c(4)), len = p))

# Diagonal averaging and recurrent forecasting
component <- cbind(r$Trend, r$seasonal, r$season2)
diagonal_averaging <- rowSums(component)
forecast <- rforecast(s, groups = list(Trend = c(1), seasonal = c(2, 3), season2 = c(4)), len = p)

# Forecast accuracy
residual <- outsample - forecast.results
# Complete SSA forecast
complete_forecast <- rforecast(ssa_data.ts, groups = list(Trend = c(1), seasonal = c(2, 3), season2 = c(4)), len = p)

# Forecast accuracy
residuals <- outsample - complete_forecast
accuracy_metrics <- accuracy(complete_forecast, outsample)

# Print accuracy metrics
print(accuracy_metrics)

# Plot actual vs forecasted values
plot(ssa_data.ts, main = "SSA Forecast vs Actual Data", col = "black", ylab = "Values", xlab = "Time")
lines(forecast.results, col = "red", lty = 2)
legend("topright", legend = c("Actual Data", "SSA Forecast"), col = c("black", "red"), lty = c(1, 2))

# Forecast future values using the best ARIMA model and SSA
forecast_arima <- forecast(BestModel, h = 12)
autoplot(forecast_arima) + ggtitle("ARIMA Forecast")

# Prophet forecasting
# Prepare the data for Prophet
ssa_data_prophet <- data.frame(ds = seq(as.Date("2000/1/1"), by = "month", length.out = length(ssa_data.ts)), y = as.numeric(ssa_data.ts))

# Fit Prophet model
m <- prophet(ssa_data_prophet)

# Make future dataframe and predict
future <- make_future_dataframe(m, periods = 12, freq = "month")
forecast_prophet <- predict(m, future)

# Plot Prophet forecast
prophet_plot_components(m, forecast_prophet)

# Combine plots of ARIMA, SSA, and Prophet forecasts for comparison
par(mfrow = c(2, 2))
plot(Crude, main = "Original Data", col = "black")
autoplot(forecast_arima) + ggtitle("ARIMA Forecast")
plot(ssa_data.ts, main = "SSA Forecast", col = "black")
lines(complete_forecast, col = "red", lty = 2)
prophet_plot_components(m, forecast_prophet)

# Reset plotting layout
par(mfrow = c(1, 1))

# Summary and conclusions
print("ARIMA Model Summary")
summary(BestModel)
print("SSA Forecast Accuracy")
print(accuracy_metrics)
print("Prophet Model Components")
prophet_plot_components(m, forecast_prophet)


