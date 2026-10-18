
#using neuralnet---------------------
install.packages("neuralnet")
library(neuralnet)
# Load the neuralnet package
library(neuralnet)

# Generate synthetic time series data
set.seed(123)
n <- 100
time <- 1:n
target <- sin(time/10) + rnorm(n)
exog1 <- rnorm(n)
exog2 <- rnorm(n)

# Create lagged features for the NARX model
lag1_target <- c(NA, head(target, -1))
lag2_target <- c(NA, NA, head(target, -2))

# Combine the data into a data frame
data <- data.frame(time, target, exog1, exog2, lag1_target, lag2_target)

# Split the data into training and testing sets
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- data[train_idx, ]
training_data <- na.omit(training_data)
testing_data <- data[test_idx, ]

# Define the NARX formula
formula <- target ~ lag1_target + lag2_target + exog1 + exog2

# Create the NARX model
model <- neuralnet(formula, data = training_data, hidden = c(5, 3),
                   act.fct = "logistic", linear.output = TRUE)

# Plot the model
plot(model)

# Train the model
trained_model <- model

# Make predictions on the test data
predictions <- predict(trained_model, testing_data)

# Evaluate the model (mean squared error)
mse <- mean((predictions - testing_data$target)^2)
cat("Mean Squared Error:", mse, "\n")



# Assuming "true_values" are the true target values and "predictions" are the predicted values
true_values <- testing_data$target  # Replace with the actual column name in your data

# Calculate the squared differences
squared_diff <- (true_values - predictions)^2

# Calculate the mean squared difference
mean_squared_diff <- mean(squared_diff)

# Calculate RMSE by taking the square root of the mean squared difference
rmse <- sqrt(mean_squared_diff)

cat("Root Mean Squared Error (RMSE):", rmse, "\n")


# Calculate the absolute differences
abs_diff <- abs(true_values - predictions)

# Calculate the mean absolute difference (MAD)
mad <- mean(abs_diff)
cat("Mean Absolute Deviation (MAD):", mad, "\n")


# Calculate the correlation between true values and predictions
correlation <- cor(true_values, predictions)

# Calculate R-squared
rsquared <- correlation^2

cat("R-squared (R²):", rsquared, "\n")



#using nnet package------------------
# Load the forecast package
library(forecast)

# Generate synthetic time series data
set.seed(123)
n <- 100
time <- 1:n
target <- sin(time/10) + rnorm(n)
exog1 <- rnorm(n)
exog2 <- rnorm(n)

# Create a time series object
ts_data <- ts(target)

# Create a data frame with exogenous inputs
data <- data.frame(ts_data, exog1, exog2)
data <-ts(data)
# Create a NARX model with neural network
narx_model <- nnetar(data, repeats = 1, size = c(5, 3))

# Make predictions
predictions <- forecast(narx_model, h = 10)

# Print the forecasted values
print(predictions)
