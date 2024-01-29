# Load libraries if not already installed
libraries <- c("caret", "maps", "mapdata", "mapproj")
libraries_to_install <- libraries[!(libraries %in% installed.packages()[,"Package"])]
if(length(libraries_to_install)) install.packages(libraries_to_install)

# Load libraries
library(caret)
library(maps)
library(mapdata)
library(mapproj)

# Load data
train_orig <- read.csv("progetto_esame/dataset/database.csv")

# Create the Timestamp column
train_orig$Timestamp <- as.POSIXct(paste(train_orig$Date, train_orig$Time), format="%m/%d/%Y %H:%M:%S")

# Remove rows with 'ValueError'
final_data <- train_orig[complete.cases(train_orig$Timestamp), ]

# Remove 'Date' and 'Time' columns
final_data <- final_data[, !(names(final_data) %in% c("Date", "Time"))]

# Explore the data
summary(final_data)

# Feature Engineering
# Example: Create new features
final_data$SquaredDepth <- final_data$Depth^2

# Remove less informative features
final_data <- final_data[, !(names(final_data) %in% c("Depth Error", "Depth Seismic Stations", "Horizontal Distance", "Magnitude Error", "Magnitude Seismic Stations", "Azimuthal Gap", "Horizontal Distance", "Horizontal Error", "Root Mean Square", "Horizontal Error"))]

# Handle missing data
# Example: Impute mean for columns with missing data
final_data$Depth[is.na(final_data$Depth)] <- mean(final_data$Depth, na.rm = TRUE)

# Remove 'Depth Error' and 'Depth Seismic Stations' columns
final_data <- final_data[, !(names(final_data) %in% c("Depth Error", "Depth Seismic Stations"))]

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

# Train the linear regression model
model_lm <- lm(Magnitude ~ Timestamp + Latitude + Longitude, data = final_data)

# Display a summary of the model
summary(model_lm)

# Split into training and test sets
set.seed(42)
split_ratio <- 0.8
indices <- createDataPartition(final_data$Magnitude, p = split_ratio, list = FALSE)

X_train <- final_data[indices, c('Timestamp', 'Latitude', 'Longitude')]
y_train <- final_data[indices, 'Magnitude']
X_test <- final_data[-indices, c('Timestamp', 'Latitude', 'Longitude')]
y_test <- final_data[-indices, 'Magnitude']

# Predict on test data
predictions <- predict(model_lm, newdata = X_test)

# Calculate the Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((predictions - y_test)^2))
cat("RMSE:", rmse, "\n")
