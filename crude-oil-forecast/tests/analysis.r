# Load libraries
libraries <- c("readr", "zoo", "tseries", "astsa", "forecast", "Rssa", "prophet", "ggplot2", "dplyr", "lubridate")
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

# Function to perform and plot decomposition
perform_decomposition <- function(ts_data) {
    decomposition <- stl(ts_data, s.window = "periodic")
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

# Define the train-test split ratio
split_ratio <- 0.8

# Calculate the index to split the data
split_index <- floor(length(diff_crude_ts) * split_ratio)

# Split the data into training and testing sets
train_diff_crude_ts <- window(diff_crude_ts, end = c(time(diff_crude_ts)[split_index]))
test_diff_crude_ts <- window(diff_crude_ts, start = c(time(diff_crude_ts)[split_index + 1]))

# Check the lengths of the splits
length(train_diff_crude_ts)
length(test_diff_crude_ts)

# ACF and PACF plots
acf2(diff_crude_ts, main = "ACF and PACF of Differenced Crude Oil Prices")

# Fit an ARIMA model
arima_model <- auto.arima(train_diff_crude_ts)
print(arima_model)

# Check the residuals
checkresiduals(arima_model)

# Forecast
forecast_arima <- forecast(arima_model, h = length(test_diff_crude_ts))
print(forecast_arima)

# Plot the forecast
autoplot(forecast_arima) + autolayer(test_diff_crude_ts, series = "Test Data")

# Forecast plot with legends
plot(forecast_arima, main = "Forecasted vs Actual Crude Oil Prices")
lines(test_diff_crude_ts, col = "red")
legend("topleft", legend = c("Forecasted", "Actual"), col = c("black", "red"), lty = 1)

# check accuracy of the model
accuracy_arima <- accuracy(forecast_arima, test_diff_crude_ts)

# Singular Spectrum Analysis (SSA) and forecasting

# Define the train-test split ratio for SSA
split_index_ssa <- floor(length(crude_ts) * split_ratio)

# Split the data into training and testing sets
train_crude_ts <- window(crude_ts, end = c(time(crude_ts)[split_index_ssa]))
test_crude_ts <- window(crude_ts, start = c(time(crude_ts)[split_index_ssa + 1]))

# Perform SSA on the training data
L <- floor(length(train_crude_ts) / 2) # Window length
ssa_obj <- ssa(train_crude_ts, L = L)

# Embedding of Henkel matrix
Z 
# Plot the singular values
plot(ssa_obj, type = "values")

# Reconstruct the series using the leading components (e.g., the first two components)
groups <- list(Trend = 1:2)
recon <- reconstruct(ssa_obj, groups = groups)

# Convert reconstructed series to a data frame for ggplot
recon_df <- data.frame(
    Date = as.Date(zoo::as.yearmon(time(train_crude_ts)), frac = 1), # Convert to 'yearmon' and then to 'Date'
    Original = as.numeric(train_crude_ts),
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
forecast_steps <- length(test_crude_ts) # Number of steps to forecast
ssa_forecast <- rforecast(ssa_obj, groups = list(Trend = 1:2), len = forecast_steps)

# Convert SSA forecast to a data frame for ggplot
ssa_forecast_df <- data.frame(
    Date = seq(forecast_start_date, by = "month", length.out = forecast_steps),
    Forecast = as.numeric(ssa_forecast)
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

# Ensure 'forecast_start_date' and 'forecast_end_date' are correctly ordered and formatted
forecast_start_date <- as.Date(zoo::as.yearmon(time(test_crude_ts)[1]), frac = 1)

forecast_end_date <- as.Date("2024-12-01")

# Adjust 'crude_ts_aligned' to fit within the actual range of 'crude_ts'
crude_ts_aligned <- window(crude_ts, start = forecast_start_date, end = forecast_end_date)

# Proceed with comparing 'ssa_forecast' with 'crude_ts_aligned'
forecast_accuracy <- accuracy(ssa_forecast, window(crude_ts, start = start(ssa_forecast_df$Date), end = end(ssa_forecast_df$Date)))
print(forecast_accuracy)

# Prophet Modeling and Forecasting
# Prepare data for Prophet
prophet_data <- crude_data %>% rename(ds = Date, y = Price)
train_start_date <- as.Date("2002-01-01")
test_start_date <- as.Date("2019-01-01")

# Set training and testing sets
train_prophet_data <- prophet_data %>% filter(ds >= train_start_date & ds < test_start_date)
test_prophet_data <- prophet_data %>% filter(ds >= test_start_date)

# Building the Prophet model
prophet_model <- prophet(train_prophet_data)
future <- make_future_dataframe(prophet_model, periods = nrow(test_prophet_data), freq = "month")
forecast_prophet <- predict(prophet_model, future)

# Prophet model accuracy
actual <- test_prophet_data$y
forecast_values <- forecast_prophet$yhat[forecast_prophet$ds >= test_start_date]
forecast_accuracy_prophet <- accuracy(forecast_values, actual)
print(forecast_accuracy_prophet)
