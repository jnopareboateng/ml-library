# Install and load necessary packages
install.packages(c("tseries", "forecast", "urca", "astsa"))
library(tseries)
library(forecast)
library(urca)
library(astsa)

# Load data
data_path <- "Modified Data.csv"  # Replace with your file path
data <- read.csv(data_path)
str(data)

# Convert to time series
price_ts <- ts(data = data$Price, start = c(2002, 1), frequency = 12)
str(price_ts)

# Plot time series
plot(price_ts)
autoplot(price_ts)

# Decompose time series
plot(decompose(price_ts))
seasonplot(price_ts, col = 1:20, pch = 19)

# Test for normality
shapiro.test(price_ts)

# Test for stationarity
ndiffs(price_ts)
nsdiffs(price_ts)

hist(price_ts)
acf(price_ts)
pacf(price_ts)
adf_price <- adf.test(price_ts)
print(adf_price)

# Differencing data
differenced_price <- diff(price_ts)
adf_differenced <- adf.test(differenced_price)

# Autocorrelation of differenced data
acf2(differenced_price)
acf(differenced_price, lag = 300)

# Fit ARIMA model
model <- auto.arima(differenced_price, trace = T)
arima_model <- arima(differenced_price, order = c(1, 1, 0))
residuals <- residuals(model)
plot(residuals)
checkresiduals(model)

# Forecast
forecasted <- forecast(arima_model, h = 24)
plot(forecasted)
sarima.for(differenced_price, 10, 1, 1)

# Check accuracy
accuracy(forecasted)

# Plot histogram
hist(price_ts, xlab = "Price", freq = F, col = "Blue", main = "Histogram of Commodity Price")
lines(density(price_ts), col = "green", lwd = 7)

# Calculate slope and intercept
price_times <- as.numeric(time(price_ts))
price_values <- as.numeric(price_ts)
SSxx <- sum((price_times - mean(price_times)) * (price_times - mean(price_times)))
SSxy <- sum((price_values - mean(price_values)) * (price_times - mean(price_times)))
slope <- SSxy / SSxx
intercept <- mean(price_values) - slope * mean(price_times)

