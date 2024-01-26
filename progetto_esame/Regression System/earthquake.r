reticulate::use_python("/Library/Developer/CommandLineTools/usr/bin/python3")

# Install libraries if not already installed
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages("randomForest")
}

if (!requireNamespace("maps", quietly = TRUE)) {
  install.packages("maps")
}

if (!requireNamespace("mapdata", quietly = TRUE)) {
  install.packages("mapdata")
}

if (!requireNamespace("mapproj", quietly = TRUE)) {
  install.packages("mapproj")
}

if (!requireNamespace("keras", quietly = TRUE)) {
  install.packages("keras")
}

# Load necessary libraries
library(caret)
library(dplyr)
library(maps)
library(mapdata)
library(mapproj)
library(keras)
library(installr)
library(nnet)

# Load CSV file
train_orig <- read.csv("progetto_esame/dataset/database.csv")

# Create Timestamp column
timestamp <- vector("numeric", length = nrow(train_orig))
for (i in 1:nrow(train_orig)) {
  tryCatch({
    ts <- as.POSIXct(strptime(paste(train_orig$Date[i], train_orig$Time[i]), format="%m/%d/%Y %H:%M:%S"))
    timestamp[i] <- as.numeric(ts)
  }, error = function(e) {
    timestamp[i] <- NA
  })
}

# Add Timestamp column to the dataframe
train_orig$Timestamp <- timestamp

# Remove rows with 'ValueError'
final_data <- train_orig[!is.na(train_orig$Timestamp) & train_orig$Timestamp != 'ValueError', ]

# Remove 'Date' and 'Time' columns
final_data <- final_data[, !(names(final_data) %in% c("Date", "Time"))]

# Display the first rows of the final dataframe
print(head(final_data))

# Create a map without specifying the projection
m <- map_data("world")

# Extract coordinates
longitudes <- final_data$Longitude
latitudes <- final_data$Latitude

# Transform coordinates
xy <- mapproject(longitudes, latitudes)

# Set image proportions
par(pin = c(10 , 5))

# Create a plot with a wider range on the x-axis
plot(xy, col = 'blue', pch = 16, cex = 0.5, main = "All affected areas", xlim = c(-180, 180))

# Add the world map
map("world", add = TRUE, col = "black", fill = FALSE)

# Create X and y dataframe
X <- final_data[c('Timestamp', 'Latitude', 'Longitude')]
y <- final_data[c('Magnitude', 'Depth')]

# Extract only necessary output columns
y <- final_data[, c('Magnitude', 'Depth')]

# Split data into training and test sets
set.seed(42)  # Set seed for reproducibility
split_ratio <- 0.8
indices <- createDataPartition(y$Magnitude, p = split_ratio, list = FALSE)

X_train <- X[indices, ]
y_train <- y[indices, ]
X_test <- X[-indices, ]
y_test <- y[-indices, ]

# Display dimensions of training and test sets
cat("Training set dimensions:", dim(X_train), dim(y_train), "\n")
cat("Test set dimensions:", dim(X_test), dim(y_test), "\n")

# Define the threshold for classification
soglia <- 5.0

# Modify the output variable for classification
y_train_class <- ifelse(y_train$Magnitude >= soglia, "High", "Low")
y_test_class <- ifelse(y_test$Magnitude >= soglia, "High", "Low")

# Modify the create_model function
create_model <- function(size, decay, activation, optimizer, loss, ...) {
  model <- nnet::nnet(
    Magnitude ~ Timestamp + Latitude + Longitude,
    data = cbind(X_train, y_train),  # Combine input and output data for training
    size = size,  # Size of the hidden layer
    decay = decay,  # Decay parameter
    linout = TRUE,  # Use a linear activation function for output
    trace = FALSE
  )

  return(model)
}

# Specify grid parameters for regression
param_grid_nnet <- expand.grid(
  size = c(5, 10, 15),  # Example values for the size of the hidden layer
  decay = c(0.001, 0.01, 0.1),  # Example values for the decay parameter
  activation = activation,
  optimizer = optimizer,
  loss = loss
)

neurons <- 10
batch_size <- 32
epochs <- 50
activation <- "tanh"
optimizer <- "adam"
loss <- "mean_squared_error"

# Modify the target variable to be numeric
y_train <- final_data[indices, c('Magnitude', 'Depth')]
y_test <- final_data[-indices, c('Magnitude', 'Depth')]

# Remove rows with missing values in the target variable
complete_rows <- complete.cases(y_train)
X_train <- X_train[complete_rows, ]
y_train <- y_train[complete_rows, ]

summary(y_train$Magnitude)

# Check data types of relevant columns
str(final_data)

# Create a GridSearchCV object with the caret library for classification
grid_regressor <- caret::train(
  x = as.data.frame(X_train),
  y = y_train$Magnitude,
  method = "nnet",
  trControl = trainControl(method = 'cv', number = 5),
  metric = 'RMSE',  # Use RMSE for regression
  tuneGrid = expand.grid(size = c(5, 10, 15), decay = c(0.001, 0.01, 0.1))
)

# Predict values based on the test set
predictions <- predict(grid_regressor, newdata = as.data.frame(X_test))

# Calculate the Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((predictions - y_test$Magnitude)^2))
cat("RMSE:", rmse, "\n")