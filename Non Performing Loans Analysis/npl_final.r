# Load libraries
library(readxl)
library(ggplot2)
library(reshape2)
library(MASS)
library(performance)
library(see)

# Load data
Our_DATA <- read_excel("D:/DEV WORK/Data Science Library/ml-library/NPL/PROJECT.xlsx")

# Summary of data
summary(Our_DATA)

# Transform NPL to log scale
Our_DATA$NPL_log <- log(Our_DATA$NPL)

# Correlation matrix
cor_matrix <- cor(Our_DATA[, c("NPL", "Lrate", "InfRate", "CAR", "Real_Growth", "Exc_Rate")])

# Visualize correlation matrix
cor_matrix_melt <- melt(cor_matrix)
ggplot(cor_matrix_melt, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
    theme_minimal()

# Multiple Linear Regression
model <- lm(NPL ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate, data = Our_DATA)
summary(model)

# Check normality of NPL
hist(Our_DATA$NPL, main = "Histogram of NPL", xlab = "NPL", col = "lightblue", border = "black")
plot(density(Our_DATA$NPL), main = "Density Plot of NPL", xlab = "NPL", col = "blue")
shapiro.test(residuals(model))

# Gamma Regression
model_gamma <- glm(NPL ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate,
    family = Gamma(link = "log"), data = Our_DATA
)
summary(model_gamma)

# Check assumptions of Gamma Regression
plot(model_gamma, which = 1)
shapiro.test(residuals(model_gamma))

# Lognormal Regression
model_lognormal <- glm(NPL_log ~ Lrate + InfRate + CAR + Real_Growth + Exc_Rate,
    family = gaussian(link = "log"), data = Our_DATA
)
summary(model_lognormal)

# Check assumptions of Lognormal Regression
plot(model_lognormal, which = 1)
shapiro.test(residuals(model_lognormal))

# Generalized Linear Models (GLM)
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

# Check normality of transformed NPL
shapiro.test(Our_DATA$NPL_log)

# Check assumptions of transformed model
plot(model_log, which = 1) # check for constant variance
dwtest(model_log)
