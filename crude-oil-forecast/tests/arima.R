install.packages("tseries")
install.packages("forecast")
library(tseries)
library(forecast)


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
?trace
m1 <- arima(data_ts, order = c(1, 1, 0))
m1
resd <- residuals(m1)
plot(resd)
checkresiduals(m1)
pre <- forecast(m1, h = 24)
plot(pre)
sarima.for(data_ts, 10, 1, 1)

accuracy(pre)
