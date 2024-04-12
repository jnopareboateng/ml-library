install.packages("tseries")
install.packages("forecast")
library(tseries)
library(forecast)


data <- read.csv(file.choose())
?attach
View(data)
attach(data)
data
str(data)


data_ts <- ts(data = data$Price, start = c(2002, 1), frequency = 12)
View(data_ts)
str(data_ts)
data_ts
?autoplot
plot(data_ts)
autoplot(data_ts)

plot(decompose(data_ts))
seasonplot(data_ts, col = 1:20, pch = 19)

shapiro.test(data_ts)
library(astsa)
?astsa
ndiffs(data_ts)
nsdiffs(data_ts)
hist(data_ts)
acf(data_ts)
pacf(data_ts)
adf_data <- adf.test(data_ts)
print(adf_data)
differenced_data <- diff(data_ts)
adf_differenced <- adf.test(differenced_data)

acf2(differenced_data)
acf(differenced_data, lag = 300)

model <- auto.arima(data_ts, trace = T)
m1 <- arima(data_ts, order = c(1, 1, 0))
m1
resd <- residuals(model)
plot(resd)
checkresiduals(model)
pre <- forecast(m1, h = 24)
plot(pre)
sarima.for(data_ts, 10, 1, 1)

accuracy(pre)


hist(data.ts, xlab = "My Data", freq = F, col = "Blue", main = "Histogram of Commodity Data")
density(data.ts)
lines(density(data.ts))
lines(density(data.ts), col = "green", lwd = 7)
?round
brentoil.times <- as.numeric(time(data.ts))
brentoil.values <- as.numeric(data.ts)
SSxx <- sum((brentoil.times - mean(brentoil.times)) * (brentoil.times - mean(brentoil.times)))
SSxy <- sum((brentoil.values - mean(brentoil.values)) * (brentoil.times - mean(brentoil.times)))
(slope <- SSxy / SSxx)
(intercept <- mean(brentoil.values) - slope * mean(brentoil.times))
