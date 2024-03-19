install.packages("tseries")
install.packages("forecast")
# # Install TensorFlow
# install.packages('tensorflow')

# # Install keras, which provides a high-level interface to TensorFlow
# install.packages('keras')

# # Install e1071 for Support Vector Machines
# install.packages('e1071')
library(tseries)
library(forecast)


data=read.csv(file.choose())
?attach
View(data)
attach(data)
data
str(data)


data_ts<-ts(data = data$Price, start = c(2000, 1), frequency = 12)
View(data_ts)
str(data_ts)

?autoplot
plot(data_ts)
autoplot(data_ts)

plot(decompose(data_ts))
seasonplot(data_ts,col=1:20,pch=19)

shapiro.test(data_ts)
library(astsa)
?astsa
ndiffs(data_ts)
hist(data_ts)
acf(data_ts)
pacf(data_ts)
adf_data<- adf.test(data_ts)
print(adf_data)
differenced_data<-diff(data_ts)
adf_differenced<-adf.test(differenced_data)




model=auto.arima(data_ts, trace = T)


pre=forecast(model, h=10)
plot(pre)



hist(data_ts, xlab = "My Data", freq = F, col ="Blue", main = "Histogram of Commodity Data")
density(data_ts)
lines(density(data_ts))
lines(density(data_ts), col = "green", lwd =7)
?round
brentoil.times = as.numeric( time(data_ts) )
brentoil.values = as.numeric(data_ts)
SSxx = sum( (brentoil.times - mean(brentoil.times) ) * (brentoil.times - mean(brentoil.times) ) )
SSxy = sum( (brentoil.values - mean(brentoil.values) ) * (brentoil.times - mean(brentoil.times) ) )
( slope = SSxy / SSxx )
( intercept = mean(brentoil.values) - slope*mean(brentoil.times) )


## Prophet model
library(prophet)
data_ts <- rename(df, ds = date, y = price)
model <- prophet(df)
future <- make_future_dataframe(model, periods = 10, freq = "month")
forecast <- predict(model, future)
plot(model, forecast)
