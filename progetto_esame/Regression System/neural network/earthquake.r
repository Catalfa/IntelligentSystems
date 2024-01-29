# Load libraries if not already installed
libraries <- c("caret", "randomForest", "maps", "mapdata", "mapproj", "nnet")
libraries_to_install <- libraries[!(libraries %in% installed.packages()[,"Package"])]
if (length(libraries_to_install)) install.packages(libraries_to_install)

# Load libraries
library(caret)
library(maps)
library(mapdata)
library(mapproj)
library(nnet)

# Load data
train_orig <- read.csv("progetto_esame/dataset/database.csv")

# Create the Timestamp column
train_orig$Timestamp <- as.POSIXct(paste(train_orig$Date, train_orig$Time), format="%m/%d/%Y %H:%M:%S")

# Remove rows with 'ValueError'
final_data <- train_orig[complete.cases(train_orig$Timestamp), ]

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
par(pin = c(10, 5))

# Create a plot with a wider range on the x-axis
plot(xy, col = 'blue', pch = 16, cex = 0.5, main = "All affected areas", xlim = c(-180, 180))

# Add the world map
map("world", add = TRUE, col = "black", fill = FALSE)

# Create X and y dataframe
X <- final_data[c('Timestamp', 'Latitude', 'Longitude')]
y <- final_data[c('Magnitude', 'Depth')]

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

# Modify the create_model function
create_model <- function(size, decay, activation, optimizer, loss, ...) {
  model <- nnet::nnet(
    Magnitude ~ Timestamp + Latitude + Longitude,
    data = cbind(X_train, y_train),  
    size = size,  
    decay = decay,  
    linout = TRUE,  
    trace = FALSE
  )

  return(model)
}

# Specify grid parameters for regression
param_grid_nnet <- expand.grid(
  size = c(5, 10, 15),
  decay = c(0.001, 0.01, 0.1)
)

# Other parameters
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

# Create a GridSearchCV object with the caret library for regression
grid_regressor <- caret::train(
  x = as.data.frame(X_train),
  y = y_train$Magnitude,
  method = "nnet",
  trControl = trainControl(method = 'cv', number = 5),
  metric = 'RMSE',
  tuneGrid = param_grid_nnet  # Use the specified parameter grid
)

# Predict values based on the test set
predictions <- predict(grid_regressor, newdata = as.data.frame(X_test))

# Calculate the Root Mean Squared Error (RMSE)
rmse <- caret::RMSE(predictions, y_test$Magnitude)
cat("RMSE:", rmse, "\n")
