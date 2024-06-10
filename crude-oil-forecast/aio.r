# Load required packages
library(tseries)
library(astsa)
library(forecast)
library(Rssa)
library(prophet)

# Load and preprocess data
file_path <- "data/Pre-processed Data.csv"
data <- read.csv(file_path)

# Convert Date column to date format
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")

# Set crude oil prices as time series data
ts_data <- ts(data$Price, start = c(2002, 1), frequency = 12)

# Check for stationarity using Augmented Dickey-Fuller test
adf_result <- adf.test(ts_data)
print(adf_result)

# Check for trend and seasonal differencing
trend_diff <- ndiffs(ts_data)
print(trend_diff)

# Perform differencing if needed
diff_data <- diff(ts_data)

# Autocorrelation and partial autocorrelation plots
autoplot(diff_data)

# Best SARIMA model selection using auto.arima()
best_model <- auto.arima(ts_data, seasonal = FALSE)
summary(best_model)

# Fit SARIMA models
fit1 <- sarima(ts_data, 1, 1, 0)
fit2 <- sarima(ts_data, 1, 1, 2)
fit3 <- sarima(ts_data, 1, 1, 6)
fit4 <- sarima(ts_data, 5, 1, 1)
fit5 <- sarima(ts_data, 5, 1, 2)
fit6 <- sarima(ts_data, 5, 1, 6)

# Compare models using AIC
aic_values <- c(AIC(fit1), AIC(fit2), AIC(fit3), AIC(fit4), AIC(fit5), AIC(fit6))
best_model_index <- which.min(aic_values)
cat("Best model:", names(aic_values)[best_model_index])

# Forecast using the best model
forecast <- forecast(best_model, h = 12)

# Plot the forecast
plot(forecast)

# Singular Spectrum Analysis (SSA)

# Set parameters
n <- length(ts_data)
p <- 20

insample <- ts_data [1:(n - p)] # Training data
outsample <- ts_data [(n - p + 1):n] # Test data

# Embedding
z <- as.matrix(insample)
x <- embed(z, p)
id <- 1:p
y <- rbind(id, x)
w <- as.matrix(y)
sort <- w[, order(-w [1, ])]

# Singular Value Decomposition (SVD)
trajectory <- t(sort) %*% sort
e.values <- eigen(trajectory)$values

# Grouping and Reconstruction
s <- ssa(insample, L = p, kind = "1d-ssa")

# Trend and Seasonal Components
trend <- reconstruct(s, groups = list(Trend = c(1)), len = p)
seasonal <- reconstruct(s, groups = list(Seasonal = c(2, 3)), len = p)

# Forecast each component separately
trend_forecast <- forecast(ts(trend, start = c(2002, 1), frequency = 12), h = 12)
seasonal_forecast <- forecast(ts(seasonal, start = c(2002, 1), frequency = 12), h = 12)

# Combine forecasts
combined_forecast <- trend_forecast + seasonal_forecast

# Plot the forecasts
plot(combined_forecast, main = "SSA Forecast", xlab = "Year", ylab = "Price")

# Prophet Modeling and Forecasting

# Load prophet data
prophet_data <- read.csv("TBill Data(2013-2022) - Prophet.csv")

# Building the Prophet model
prophet_model <- prophet(prophet_data)
future <- make_future_dataframe(prophet_model, periods = 12)
forecast <- predict(prophet_model, future)

# Prophet model accuracy
actual <- prophet_data$y
predicted <- forecast$yhat [-(1:365)]

mae <- mean(abs(actual - predicted))
rmse <- sqrt(mean((actual - predicted)^2))
mape <- mean(abs((actual - predicted) / actual)) * 100
smape <- mean(200 * abs(actual - predicted) / (abs(actual) + abs(predicted)))
forecast_coverage <- mean((actual >= forecast$yhat_lower) & (actual <= forecast$yhat_upper)) * 100

# Visualize forecasts
dyplot.prophet(prophet_model, forecast)
prophet_plot_components(prophet_model, forecast)
