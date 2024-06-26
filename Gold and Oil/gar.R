# Load required libraries
library(tseries)  # for GARCH model fitting
library(quantmod) # for fetching financial data


# Function to fetch commodity prices data
getCommodityPrices <- function(symbol, start_date, end_date) {
  data <- getSymbols(symbol, src = 'yahoo', from = start_date, to = end_date, auto.assign = FALSE)
  prices <- Cl(data)
  return(prices)
}

# Fetch commodity prices data for oil and gold
oil_prices <- getCommodityPrices("CL=F", start_date = "2010-01-01", end_date = "2023-12-31") # Example date range
gold_prices <- getCommodityPrices("GC=F", start_date = "2010-01-01", end_date = "2023-12-31") # Example date range

# Combine the two series into a single data frame
commodity_data <- data.frame(Oil = oil_prices, Gold = gold_prices)

# Fit GARCH model
garch_model <- garch(commodity_data[, "Oil"], order = c(1, 1), trace = FALSE)

# Summary of GARCH model
summary(garch_model)

# Plot of GARCH model
plot(garch_model)

# Forecasting with the GARCH model
forecast_garch <- predict(garch_model, n.ahead = 10) # Example: forecasting 10 periods ahead
print(forecast_garch)
