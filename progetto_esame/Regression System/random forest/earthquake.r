# Load libraries if not already installed
libraries <- c("caret", "randomForest", "maps", "mapdata", "mapproj", "nnet")
to_install <- libraries[!(libraries %in% installed.packages()[,"Package"])]
if(length(to_install)) install.packages(to_install)

# Load libraries
library(caret)
library(maps)
library(mapdata)
library(mapproj)
library(randomForest)

# Load data
train_orig <- read.csv("progetto_esame/dataset/database.csv")

# Create Timestamp column
train_orig$Timestamp <- as.POSIXct(paste(train_orig$Date, train_orig$Time), format="%m/%d/%Y %H:%M:%S")

# Remove rows with 'ValueError'
final_data <- train_orig[complete.cases(train_orig$Timestamp), ]

# Remove 'Date' and 'Time' columns
final_data <- final_data[, !(names(final_data) %in% c("Date", "Time"))]

# Explore data
summary(final_data)

# Feature Engineering
# Example: Create new feature
final_data$SquaredDepth <- final_data$Depth^2

# Remove uninformative features
final_data <- final_data[, !(names(final_data) %in% c("Depth Error", "Depth Seismic Stations", "Horizontal Distance", "Magnitude Error","Magnitude Seismic Stations","Azimuthal Gap","Horizontal Distance","Horizontal Error","Root Mean Square","Horizontal Error"))]

# Handling missing data
# Example: Impute mean for columns with missing values
final_data$Depth[is.na(final_data$Depth)] <- mean(final_data$Depth, na.rm = TRUE)

# Remove 'Depth Error' and 'Depth Seismic Stations' columns
final_data <- final_data[, !(names(final_data) %in% c("Depth Error", "Depth Seismic Stations"))]

# Create map without specifying projection
m <- map_data("world")

# Extract coordinates
longitudes <- final_data$Longitude
latitudes <- final_data$Latitude

# Transform coordinates
xy <- mapproject(longitudes, latitudes)

# Set image proportions
par(pin = c(10, 5))

# Create a plot with a wider range on the x-axis
plot(xy, col = 'blue', pch = 16, cex = 0.5, main = "All Areas of Interest", xlim = c(-180, 180))

# Add world map
map("world", add = TRUE, col = "black", fill = FALSE)

# Train Random Forest model
model_rf <- randomForest(Magnitude ~ Timestamp + Latitude + Longitude, data = final_data)

# Split into training and test sets
set.seed(42)
split_ratio <- 0.8
indices <- createDataPartition(final_data$Magnitude, p = split_ratio, list = FALSE)

X_train <- final_data[indices, c('Timestamp', 'Latitude', 'Longitude')]
y_train <- final_data[indices, 'Magnitude']
X_test <- final_data[-indices, c('Timestamp', 'Latitude', 'Longitude')]
y_test <- final_data[-indices, 'Magnitude']

# Predict on test data
predictions <- predict(model_rf, newdata = X_test)

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((predictions - y_test)^2))
cat("RMSE:", rmse, "\n")
