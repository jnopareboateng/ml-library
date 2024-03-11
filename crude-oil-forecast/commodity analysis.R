install.packages("tseries")
install.packages("forecast")
library(tseries)
library(forecast)


data=read.csv(file.choose())
?attach
View(data)
attach(data)
data
str(data)


data.ts<-ts(data = data$Price, start = c(2000, 1), frequency = 12)
View(data.ts)
str(data.ts)

?autoplot
plot(data.ts)
autoplot(data.ts)

plot(decompose(data.ts))
seasonplot(data.ts,col=1:20,pch=19)

shapiro.test(data.ts)
library(astsa)
?astsa
ndiffs(data.ts)
hist(data.ts)
acf(data.ts)
pacf(data.ts)
adf_data<- adf.test(data.ts)
print(adf_data)
differenced_data<-diff(data.ts)
adf_differenced<-adf.test(differenced_data)




model=auto.arima(data.ts, trace = T)


pre=forecast(model, h=10)
plot(pre)



hist(data.ts, xlab = "My Data", freq = F, col ="Blue", main = "Histogram of Commodity Data")
density(data.ts)
lines(density(data.ts))
lines(density(data.ts), col = "green", lwd =7)
?round
brentoil.times = as.numeric( time(data.ts) )
brentoil.values = as.numeric(data.ts)
SSxx = sum( (brentoil.times - mean(brentoil.times) ) * (brentoil.times - mean(brentoil.times) ) )
SSxy = sum( (brentoil.values - mean(brentoil.values) ) * (brentoil.times - mean(brentoil.times) ) )
( slope = SSxy / SSxx )
( intercept = mean(brentoil.values) - slope*mean(brentoil.times) )
