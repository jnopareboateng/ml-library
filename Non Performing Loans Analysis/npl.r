# Load the package
library(performance)


library(readxl)
Our_DATA <- read_excel("D:/DEV WORK/Data Science Library/ml-library/NPL/PROJECT.xlsx")
summary(Our_DATA)

Our_DATA$NPL_log <- log(Our_DATA$NPL)

cor_matrix <- cor(Our_DATA[, c("NPL", "Lrate", "InfRate", "CAR", "Real_Growth", "Exc_Rate")])

library(ggplot2)
library(reshape2)
install.packages("reshape2")
cor_matrix_melt <- melt(cor_matrix)
ggplot(cor_matrix_melt, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
    theme_minimal()

# Multiple Linear Regression
model <- lm(NPL ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate, data = Our_DATA)
summary(model)

hist(Our_DATA$NPL, main = "Histogram of NPL", xlab = "NPL", col = "lightblue", border = "black")

plot(density(Our_DATA$NPL), main = "Density Plot of NPL", xlab = "NPL", col = "blue")
shapiro.test(residuals(model))

# Gamma REgression
library(MASS)
model_gamma <- glm(NPL ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate,
    family = Gamma(link = "log"), data = Our_DATA
)
summary(model_gamma)

plot(model_gamma, which = 1)
shapiro.test(residuals(model_gamma))

# Lognormal Regression
model_lognormal <- glm(NPL_log ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate,
    family = gaussian(link = "log"), data = Our_DATA
)
summary(model_lognormal)

plot(model_lognormal, which = 1)
shapiro.test(residuals(model_lognormal))

# Generalized Linear Models (GLM)
library(MASS)
model_ig <- glm(NPL ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate,
    family = inverse.gaussian(link = "log"), data = Our_DATA
)

model_weibull <- glm(NPL ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate,
    family = weibull(link = "log"), data = Our_DATA
)

summary(model_ig)
plot(model_ig, which = 1) # deviance plot
plot(model_ig, which = 2) # pearson residuals
dwtest(model_ig)
summary(model_weibull)

# Transformation of NPL to Normal Distribution
Our_DATA$NPL_log <- log(Our_DATA$NPL)
model_log <- lm(NPL_log ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate, data = Our_DATA)
summary(model_log)
shapiro.test(Our_DATA$NPL_log) # check normality
plot(model_log, which = 1) # check for constant variance
dwtest(model_log)

check_model(model_log, test = "normality")
