# Load libraries
libraries <- c("readr", "zoo", "tseries", "astsa", "forecast", "Rssa", "tidyverse", "prophet", "ggplot2", "dplyr", "lubridate")
install_if_missing <- function(libraries) {
    for (library in libraries) {
        if (!require(library, character.only = TRUE)) {
            install.packages(library, dependencies = TRUE)
            library(library, character.only = TRUE)
        }
    }
}
install_if_missing(libraries)

# Load data
crude_data <- read_csv(file.choose())

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

# Function to perform and plot decomposition
perform_decomposition <- function(crude_ts) {
    decomposition <- stl(crude_ts, s.window = "periodic")
    plot(decomposition)
    return(decomposition)
}

decomposition <- perform_decomposition(crude_ts)

# Perform the Augmented Dickey-Fuller (ADF) test
adf_test <- adf.test(crude_ts)
print(adf_test)

# Determine the number of differences required
d <- ndiffs(crude_ts)
print(paste("Number of differences required:", d))

# Difference the series
diff_crude_ts <- diff(crude_ts, differences = d)

# Plot the differenced series
autoplot(diff_crude_ts)

# Perform ADF test on differenced series
adf_test_differenced <- adf.test(diff_crude_ts)
print(adf_test_differenced)

# Decompose the differenced series
decomposition_diff <- perform_decomposition(diff_crude_ts)

# Split the data into training and testing sets
train_data <- window(crude_ts, end = c(2018, 12))
test_data <- window(crude_ts, start = c(2019, 1))

# Check the lengths of the splits
length(train_data)
length(test_data)

# ACF and PACF plots
acf2(diff_crude_ts, main = "ACF and PACF of Differenced Crude Oil Prices")

# Fitting competing models
sarima(diff_crude_ts, 1, 0, 0) # $AIC[1] 6.275165 $AICc [1] 6.275358 $BIC [1] 6.275358

sarima(diff_crude_ts, 6, 0, 0) # $AIC[1] 6.286514 $AICc [1] 6.28835 $BIC [1] 6.398879

sarima(diff_crude_ts, 0, 0, 1) # $AIC[1] 6.29456 $AICc [1] 6.294753 $BIC [1] 6.336697

sarima(diff_crude_ts, 0, 0, 6) # $AIC [1] 6.283195 $AICc [1] 6.285031 $BIC [1] 6.39556

sarima(diff_crude_ts, 1, 0, 1) # $AIC [1] 6.282598 $AICc [1] 6.282985 $BIC [1] 6.33878

sarima(diff_crude_ts, 1, 0, 6) # $AIC [1] 6.288729 $AICc [1] 6.291099 $BIC [1] 6.415139

# Fit an ARIMA model
arima_model <- auto.arima(train_data, d = d, seasonal = FALSE, stepwise = TRUE)
print(arima_model)

# Check the residuals
checkresiduals(arima_model)

# Predict on the Test data
forecast_test <- forecast(arima_model, h = length(test_data))
print(forecast_test)

# Plot the forecast
autoplot(forecast_test) + autolayer(test_data, series = "Test Data")

forecast_full <- forecast(auto.arima(crude_ts, d = d), h = 24)

# Plot the full forecast
autoplot(forecast_full) + autolayer(crude_ts, series = "Crude Oil Prices")

# Print full forecasted prices
print(forecast_full)

# check accuracy of the model
accuracy_arima <- accuracy(forecast_test$mean, test_data)
print(accuracy_arima)

# Singular Spectrum Analysis (SSA) and forecasting

# Perform SSA on the training data
L <- floor(length(crude_ts) / 5) # Window length
ssa <- ssa(train_data, L = L)

# Plot eigenvalues
plot(ssa, type = "values", main = "Eigenvalues")

# Plot eigenvectors
plot(ssa, type = "vectors", main = "Eigenvectors")


# Plot the original series
plot(train_data, main = "Original Series")

# Reconstruct components
rec <- reconstruct(ssa, groups = list(trend = 1:2, seasonal = 3:6))

# Plot the decomposition for the training data
plot(rec)

# Plot correlation matrix
wcor <- wcor(ssa, groups = list(Trend = c(1:2), seasonal = c(3:6), noise = c(7:10)))
plot(wcor, main = "W-correlation matrix")

# Extract individual components
trend <- rec$trend
seasonal <- rec$seasonal
residuals <- train_data - trend - seasonal

# Plot individual components
par(mfrow = c(4, 1))
plot(train_data, main = "Original Training Series")
plot(trend, main = "Trend")
plot(seasonal, main = "Seasonal")
plot(residuals, main = "Residuals")

# Forecast for the test period
predicted <- forecast(ssa, groups = list(1:12), len = length(test_data), only.new = TRUE)

# Plot forecast vs actual test data

plot(predicted, col = "blue")
lines(test_data, col = "red")
legend("topleft", legend = c("Predicted", "Actual"), col = c("blue", "red"), lty = 1)

# forecast 24 months into the future
ssa_full <- ssa(crude_ts, L = L)
forecast_future <- forecast(ssa_full, groups = list(1:12), len = 24, only.new = TRUE)

# Plot the forecast
plot(forecast_future)

# Calculate forecast accuracy
accuracy_ssa <- accuracy(predicted$mean, test_data)
print(accuracy_ssa)


# Prophet Modeling and Forecasting
# Prepare data for Prophet
prophet_data <- crude_data %>% rename(ds = Date, y = Price)
train_start_date <- as.Date("2002-01-01")
test_start_date <- as.Date("2019-01-01")

# Set training and testing sets
train_data <- prophet_data %>% filter(ds >= train_start_date & ds < test_start_date)
test_data <- prophet_data %>% filter(ds >= test_start_date)

length(train_data$y)
length(test_data$y)
# Building the Prophet model
prophet_model <- prophet(train_data)
future <- make_future_dataframe(prophet_model, periods = nrow(test_data), freq = "month")
forecast_prophet <- predict(prophet_model, future)

# Prophet model accuracy
actual <- test_data$y
predicted_values <- forecast_prophet$yhat[forecast_prophet$ds >= test_start_date]
predicted_accuracy <- accuracy(predicted_values, actual)
print(predicted_accuracy)

# Forecast into the future for the next 24 months after the last data point in the test set
future_forecast <- make_future_dataframe(prophet_model, periods = 24, freq = "month")
forecast_prophet_future <- predict(prophet_model, future_forecast)

# Plot the Prophet forecast
plot(prophet_model, forecast_prophet_future)

# Plot model diagnostics
prophet_plot_components(prophet_model, forecast_prophet_future)

