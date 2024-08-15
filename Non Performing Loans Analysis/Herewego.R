library(readxl)
library(effects)
library(see)
library(lessR)
library(performance)
library(patchwork)
library(lmtest)
library(car)
library(GGally)
library(tseries)
library(ggplot2)
Our_DATA <-read_excel("C:/Users/USER/Desktop/Project7/PROJECT.xlsx")
View(Our_DATA)
## fitting the LRM
model<-lm(NPL~Lrate+InfRate+CAR+Real_Growth+Exc_Rate, data=Our_DATA)
summary(model)

## Alternative approach 
Regression(NPL~Lrate+InfRate+CAR+Real_Growth+Exc_Rate, data=Our_DATA)

#Checking effect of each determinant's effect on NPL
?effects
?allEffects
plot(allEffects(model))

## checking for the assumptions of MLR
?check_model
check_model(model)

# Test for the assumptions
##1. Homoscedasticity using
##Breusch-Pagan Test
? bptest()
bptest(model, varformula = NULL, studentize = TRUE, data =Our_DATA, weights = NULL)
## Score test
??scoretest
ncvTest(model)
### Linearity test
??rainbowTest
raintest(model)

#. Independence/ no autocorrelation test using Durbin-Watson Test
?durbinWatsonTest()
durbinWatsonTest(model)


## Normality of residual test
plot(model, which=1)
qqnorm(residuals(model))

### gaph of the correlation matrix
??ggpairs
data <-Our_DATA[ , c("NPL","Lrate","InfRate","Real_Growth","CAR","Exc_Rate")]
ggpairs(data)

##############################################################
## Graphing each determinant using time series

########## Non Performing Loan Graph ##################
NPLdata <- ts(Our_DATA$NPL, start=c(2016, 1),end=c(2022, 12), frequency=12)
tsp = attributes(NPLdata)$tsp
dates = seq(as.Date("2016-01-01"), by = "month", along =NPLdata )
plot(NPLdata, xaxt = "n", main= "Non Performing Loan ratio at Bank of Ghana from 2016-2022",ylab="NPL", xlab="time")
axis(1, at = seq(tsp[1], tsp[2], along = NPLdata), labels = format(dates, "%Y-%m"))

######## Lending Rate Graph ###############

Lratedata <- ts(Our_DATA$Lrate, start=c(2016, 1),end=c(2022, 12), frequency=12)
tsp = attributes(Lratedata)$tsp
dates = seq(as.Date("2016-01-01"), by = "month", along =Lratedata )
plot(Lratedata, xaxt = "n", main= "Average Bank Lending Rate at Bank of Ghana from 2016-2022",ylab="Lending Rate", xlab="time")
axis(1, at = seq(tsp[1], tsp[2], along = Lratedata), labels = format(dates, "%Y-%m"))

######## Inflation Rate #######################
Infdata <- ts(Our_DATA$InfRate, start=c(2016, 1),end=c(2022, 12), frequency=12)
tsp = attributes(Infdata)$tsp
dates = seq(as.Date("2016-01-01"), by = "month", along =Infdata )
plot(Infdata, xaxt = "n", main= "Inflation Rate from 2016-2022 in Ghana",ylab="Inflation Rate", xlab="time")
axis(1, at = seq(tsp[1], tsp[2], along = Infdata), labels = format(dates, "%Y-%m"))

########### Exchange Rate ###############
Excdata <- ts(Our_DATA$Exc_Rate, start=c(2016, 1),end=c(2022, 12), frequency=12)
tsp = attributes(Excdata)$tsp
dates = seq(as.Date("2016-01-01"), by = "month", along =Excdata )
plot(Excdata, xaxt = "n", main= "Exchange Rate (GHs/USD) from 2016-2022 in Ghana",ylab="Exchange Rate", xlab="time")
axis(1, at = seq(tsp[1], tsp[2], along = Excdata), labels = format(dates, "%Y-%m"))

########### Capital adequacy Ratio ###########
CARdata <- ts(Our_DATA$CAR, start=c(2016, 1),end=c(2022, 12), frequency=12)
tsp = attributes(CARdata)$tsp
dates = seq(as.Date("2016-01-01"), by = "month", along =CARdata )
plot(CARdata, xaxt = "n", main= "Bank of Ghana Capital Adequacy Ratio from 2016-2022",ylab="Capital Adequacy Ratio", xlab="time")
axis(1, at = seq(tsp[1], tsp[2], along = CARdata), labels = format(dates, "%Y-%m"))

######### Real Growth Rate #############################
Realdata <- ts(Our_DATA$Real_Growth, start=c(2016, 1),end=c(2022, 12), frequency=12)
tsp = attributes(Realdata)$tsp
dates = seq(as.Date("2016-01-01"), by = "month", along =Realdata )
plot(Realdata, xaxt = "n", main= "Bank of Ghana Real Growth Rate from 2016-2022",ylab="Real Growth Rate", xlab="time")
axis(1, at = seq(tsp[1], tsp[2], along = Realdata), labels = format(dates, "%Y-%m"))
     
     