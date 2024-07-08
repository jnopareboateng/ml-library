crude <- c(
  19.96, 20.19, 24.03, 26.03, 25.69, 24.51, 25.67, 26.26, 28.32,
  27.51, 24.5, 27.45, 30.18, 32.36, 29.45, 24.79, 25.37, 27.16,
  28.21, 29.4, 26.78, 28.97, 28.83, 29.58, 30.56, 30.34, 32.73,
  30, 37.13, 35.52, 37.67, 41.73, 42.82, 49.38, 44.55, 40.64,
  44.88, 45.85, 53.28, 53.22, 49.85, 55.6, 57.93, 63.84, 63.72,
  59.44, 56.21, 57.61, 63.86, 61.1, 63.06, 70.56, 70.97, 69.74,
  74.24, 73.87, 63.49, 60.13, 60, 62.54, 54.56, 58.96, 62.36,
  67.49, 67.92, 70.55, 75.84, 71.17, 77, 82.47, 92.06, 91.51,
  91.92, 94.49, 102.98, 110.43, 124.61, 133.47, 134.79, 115.22,
  100.75, 73.6, 55.05, 43.29, 45.62, 43.73, 47.32, 51.23, 58.57,
  69.34, 65.76, 73.07, 68.19, 73.87, 77.5, 75.24, 76.92, 74.75,
  79.9, 85.68, 76.99, 75.66, 75.49, 77.11, 78.21, 83.49, 86.11,
  92.34, 96.82, 104.09, 114.62, 123.13, 114.53, 113.91, 116.68,
  109.82, 109.96, 108.8, 110.61, 107.72, 111.63, 119.15, 124.62,
  120.37, 109.36, 95.89, 102.77, 113.19, 113.04, 111.52, 109.53,
  109.19, 112.28, 116.11, 109.53, 103.31, 103.32, 103.3, 107.37,
  110.25, 111.21, 109.45, 107.77, 110.6, 107.32, 108.8, 107.68,
  108.1, 109.2, 111.97, 108.21, 103.48, 98.56, 88.07, 79.48,
  62.36, 49.77, 58.7, 57.01, 60.9, 65.62, 63.75, 56.75, 48.18,
  48.57, 49.12, 45.72, 38.92, 31.93, 33.44, 39.8, 43.34, 47.63,
  49.89, 46.58, 47.16, 47.23, 51.42, 47.08, 54.93, 55.51, 55.98,
  52.53, 53.72, 51.11, 47.54, 49.2, 51.87, 55.23, 57.47, 62.87,
  64.27, 69.09, 65.7, 66.68, 71.67, 77.06, 75.94, 75.04, 73.85,
  79.09, 80.63, 65.96, 57.67, 60.23, 64.5, 67.05, 71.66, 70.3,
  63.05, 64.19, 59.47, 62.29, 59.63, 62.71, 65.17, 63.67, 55.53,
  33.73, 26.63, 32.11, 40.77, 43.24, 45.04, 41.87, 41.36, 43.98,
  50.23, 55.33, 62.27, 65.84, 65.33, 68.34, 73.35, 74.29, 70.51,
  74.88, 83.75, 80.75, 74.8, 85.48, 94.28, 112.51, 105.81, 111.55,
  117.22, 105.14, 97.74, 90.57, 93.6, 90.38, 81.34
)


length(crude)

Crude <- ts(crude, start = 2002, frequency = 12)

plot(Crude, main = "A Plot of Total Deliveries", las = 1)


# Packages Required

require(tseries)
require(astsa)
require(forecast)

adf.test(Crude)

# Checking Trend Differencing and Seasonal Differencing
ndiffs(Crude)

nsdiffs(Crude) # seasonal differencing


adf.test(Crude)

dCrude <- diff(Crude)


adf.test(dCrude)

autoplot(dCrude)



# SARIMA Model Building

acf2(dCrude)


BestModel <- auto.arima(Crude, seasonal = FALSE)

BestModel

summary(BestModel)




# Best Model Selection

fit1 <- sarima(Crude, 1, 1, 0)

fit2 <- sarima(Crude, 1, 1, 2)

fit3 <- sarima(Crude, 1, 1, 6)


fit4 <- sarima(Crude, 5, 1, 1)

fit5 <- sarima(Crude, 5, 1, 2)

fit6 <- sarima(Crude, 5, 1, 6)








# Since Fit 9 has the lowest AIC value, it is obtained as the best model in forcasting the
#      182-Day Treasury Bill rate.


# Forecasting using the Best Model on a Quarterly Basis

sarima.for(Crude, 3, 1, 1, 0)

sarima.for(Crude, 6, 1, 1, 0)

sarima.for(Crude, 9, 1, 1, 0)

sarima.for(Crude, 12, 1, 1, 0)



# ARIMA Predicted values for crude oil prices 2023
############################################################################################
# forecast2024 = c(78.02410, 76.89465, 76.60036, 76.62508, 76.77165, 76.96477, 77.17566,
# 77.39335, 77.61363, 77.83490, 78.05655, 78.27835)
############################################################################################




rm(list = ls())
############################################################################################
############################################################################################
############################################################################################


#########################################################
#     SINGULAR SPECTRUM MODELING AND FORECASTING        #
#########################################################

# LOADING THE NECCESSARY PACKAGES
require(Rssa)


# Reading SSA data into R. Data must have just a column.
# Data is named ("ssa_data")
ssa_data <- read.csv(file.choose(), sep = ",", h = T)

ssa_data.ts <- ts(ssa_data, start = c(2000, 1), frequency = 12)
plot(ssa_data.ts)

n <- length(ssa_data.ts) # Entire Dataset
p <- 20


insample <- ssa_data.ts[1:(n - p)] # Training Dataset
n1 <- length(insample)
n1

outsample <- ssa_data.ts[(n - p + 1):n] # Test Dataset
head(outsample)



# Parameter window length(L)
L <- n1 / 2 # Trial and error (L=<N/2)
K <- n1 - L + 1


## EMBEDDING
z <- as.matrix(insample)
z

x <- embed(z, L)
id <- 1:L

y <- rbind(id, x)
y


w <- as.matrix(y)
sort <- w[, order(-w[1, ])]

THankel <- as.matrix(sort[-1, (1:L)])

Hankel <- t(THankel)


#### SVD - Singular Value Decomposition##
trajectory <- Hankel %*% t(Hankel)
dim(trajectory)


# eigen value
e.value <- eigen(trajectory)$values
e.value

total <- sum(e.value)
propv <- e.value / total
kumv <- cumsum(propv)


# cumulative variance explained from an eigenvector
kumv

# eigen vector
e.vector <- eigen(trajectory)$vectors
e.vector

# plot Singular value
s <- ssa(insample, L = L, kind = "1d-ssa")
s

# plot tree
plot(s)

# plot eigen vector 1D
plot(s, type = "vectors", plot.method = "matplot", idx = 1:10)


# plot eigen vector 1D
plot(s, type = "paired", idx = 1:(10)) # (1:L-1)


# 2 GROUPING #####
r <- reconstruct(s, groups = list(Trend = c(1), seasonal = c(2, 4), season2 = c(5)), len = p)
r

plot(wcor(s, groups = list(Trend = c(1), seasonal = c(2, 4), season2 = c(5)), len = p))


## Diagonal Averaging
component <- cbind(r$Trend, r$seasonal, r$season2)

diagonal.averaging <- rowSums(component)
diagonal.averaging


## Reccurent Forecasting
forecast <- rforecast(s, groups = list(Trend = c(1), seasonal = c(2, 4), season2 = c(5)), len = p)
forecast

# Forecast results
forecast.results <- as.matrix(forecast$Trend + forecast$seasonal + forecast$season2)
forecast.results


# Forecasting Accuracy
residual <- outsample - forecast.results
residual



## cumulative variance of the data explained from the eigenvector, then compositional grouping in

s.complete <- ssa(ssa_data.ts, L = L, kind = "1d-ssa")
r.complete <- reconstruct(s.complete, groups = list(Trend = c(1), seasonal = c(2, 4), season2 = c(5)), len = 12) # Forcasting 12 months ahead.
component.complete <- cbind(r.complete$Trend, r.complete$seasonal, r.complete$season2)
diag.avr.complete <- rowSums(component.complete)

complete.forecast <- rforecast(s.complete, groups = list(Trend = c(1), seasonal = c(2, 4), season2 = c(5)), len = 12)

complete.forecast.results <- as.matrix(complete.forecast$Trend + complete.forecast$seasonal + complete.forecast$season2)
complete.forecast.results # forecasted values for 2023



# Plot
complete.data <- as.matrix(ssa_data.ts)
empty.comp.data <- matrix(NA, 12)
data.comp.gab <- rbind(complete.data, empty.comp.data)
dim(data.comp.gab)


# Complete Predicted Data
data.pred.comp <- as.matrix(diag.avr.complete)
data.pred.empty <- matrix(NA, 12)
data.pred.gab <- rbind(data.pred.comp, data.pred.empty)
dim(data.pred.gab)


# Data forecasting
forecast_empty <- matrix(NA, 120)
forecast_gab <- rbind(forecast_empty, complete.forecast.results)
dim(forecast_gab)

ts.plot(cbind(data.comp.gab, data.pred.gab, forecast_gab), main = paste("Actual Data, Predicted & Forecased Data using SSA"), col = c("blue", "red", "orange"))
legend("bottomleft", c("Actual", "Prediction", "Forecasting"), col = c("blue", "red", "orange"), lty = 1)




# SSA Predicted values for crude oil prices 2023
############################################################################################
# forecast2024 = c(75.97527, 76.81464,77.66009, 78.50793, 79.35447, 80.19570, 81.02751,
# 81.84568, 82.64620, 83.42504, 84.17812, 84.90167)
############################################################################################






rm(list = ls())
############################################################################################
############################################################################################
############################################################################################




#################################################
#        PROPHET MODELING AND FORECASTING       #
#################################################

# LOADING THE NECCESSARY PACKAGES
require(prophet)

# Reading prophet data into R. Data must have two columns
# with date and data points column.
# Data is named ("TBill Data(2013-2022) - Prophet")
prop_data <- read.csv(file.choose(), sep = ",", h = T)
str(prop_data)


# Building the Prophet model
prophet_model <- prophet(prop_data)
future <- make_future_dataframe(prophet_model, periods = 395) # Because prophet model's prediction is on a daily basis, hence the period.
forecast <- predict(prophet_model, future)


# Prophet model Accuracy
actual <- prop_data$y
predicted <- forecast$yhat[-(1:395)]

mae <- mean(abs(actual - predicted))
mae

rmse <- sqrt(mean((actual - predicted)^2))
rmse

mape <- mean(abs((actual - predicted) / actual)) * 100
mape

smape <- mean(200 * abs(actual - predicted) / (abs(actual) + abs(predicted)))
smape

forecast_coverage <- mean((actual >= forecast$yhat_lower) & (actual <= forecast$yhat_upper)) * 100
forecast_coverage


# Note that the prophet forecast is daily so the averages are taken in order to get it monthly.
# Saving the prophet predicted data in excel


prophet.predicted <- (forecast[c("ds", "yhat")])
write.csv(prophet.predicted, "M:/2. Research Assistant/KNUST/5. THESIS/prophet_predicted.csv", row.names = FALSE)

OR

#    Dates = forecast$ds
#    Values = forecast$yhat

write.csv(Dates, "K:/2. Research Assistant/2024/Gabriel Thesis from Harris/Crude/Dates.csv", row.names = FALSE)
write.csv(Values, "K:/2. Research Assistant/2024/Gabriel Thesis from Harris/Crude/Values.csv", row.names = FALSE)





# Visualizing your predicted points
dyplot.prophet(prophet_model, forecast)
prophet_plot_components(prophet_model, forecast)




# Prophet Predicted values for crude oil prices 2023
############################################################################################
# forecast2024 = c(47.3496871, 55.07621297, 68.20781799, 67.6089845, 58.29155949, 59.31013723,
# 60.17765653, 57.60633012, 66.1618663, 66.57049058, 63.66951509, 50.70697681, 44.72786805)
############################################################################################
