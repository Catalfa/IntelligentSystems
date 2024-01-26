reticulate::use_python("/Library/Developer/CommandLineTools/usr/bin/python3")

# Installa le librerie se non sono già installate
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


# Carica le librerie necessarie
library(caret)
library(dplyr)
library(maps)
library(mapdata)
library(mapproj)
library(keras)
library(installr)
library(nnet)


# Carica il file CSV
train_orig <- read.csv("progetto_esame/dataset/database.csv")

# Creazione della colonna Timestamp
timestamp <- vector("numeric", length = nrow(train_orig))
for (i in 1:nrow(train_orig)) {
  tryCatch({
    ts <- as.POSIXct(strptime(paste(train_orig$Date[i], train_orig$Time[i]), format="%m/%d/%Y %H:%M:%S"))
    timestamp[i] <- as.numeric(ts)
  }, error = function(e) {
    timestamp[i] <- NA
  })
}

# Aggiunge la colonna Timestamp al dataframe
train_orig$Timestamp <- timestamp

# Rimuovi le righe con 'ValueError'
final_data <- train_orig[!is.na(train_orig$Timestamp) & train_orig$Timestamp != 'ValueError', ]

# Rimuovi le colonne 'Date' e 'Time'
final_data <- final_data[, !(names(final_data) %in% c("Date", "Time"))]

# Mostra le prime righe del dataframe finale
print(head(final_data))

# Creazione della mappa senza specificare la proiezione
m <- map_data("world")

# Estrai le coordinate
longitudes <- final_data$Longitude
latitudes <- final_data$Latitude

# Trasforma le coordinate
xy <- mapproject(longitudes, latitudes)

# Imposta le proporzioni dell'immagine
par(pin = c(10 , 5))

# Crea il grafico con intervallo più ampio sull'asse x
plot(xy, col = 'blue', pch = 16, cex = 0.5, main = "All affected areas", xlim = c(-180, 180))

# Aggiungi la mappa del mondo
map("world", add = TRUE, col = "black", fill = FALSE)

# Creazione del dataframe X e y
X <- final_data[c('Timestamp', 'Latitude', 'Longitude')]
y <- final_data[c('Magnitude', 'Depth')]

# Estrai solo le colonne di output necessarie
y <- final_data[, c('Magnitude', 'Depth')]

# Split dei dati in training e test set
set.seed(42)  # Imposta il seed per la riproducibilità
split_ratio <- 0.8
indices <- createDataPartition(y$Magnitude, p = split_ratio, list = FALSE)

X_train <- X[indices, ]
y_train <- y[indices, ]
X_test <- X[-indices, ]
y_test <- y[-indices, ]

# Mostra le dimensioni dei set di training e test
cat("Training set dimensions:", dim(X_train), dim(y_train), "\n")
cat("Test set dimensions:", dim(X_test), dim(y_test), "\n")

# Define the threshold for classification
soglia <- 5.0

# Modify the output variable for classification
y_train_class <- ifelse(y_train$Magnitude >= soglia, "Alta", "Bassa")
y_test_class <- ifelse(y_test$Magnitude >= soglia, "Alta", "Bassa")

# Modifica la funzione create_model
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

# Specifica i parametri della griglia per la regressione
param_grid_nnet <- expand.grid(
  size = c(5, 10, 15),  # Esempio di valori per la dimensione del layer nascosto
  decay = c(0.001, 0.01, 0.1),  # Esempio di valori per il parametro di decadimento
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

# Rimuovi le righe con valori mancanti nella variabile target
complete_rows <- complete.cases(y_train)
X_train <- X_train[complete_rows, ]
y_train <- y_train[complete_rows, ]

summary(y_train$Magnitude)

# Verifica i tipi di dati delle colonne rilevanti
str(final_data)

# Creazione di un oggetto GridSearchCV con la libreria caret per la classificazione
grid_regressor <- caret::train(
  x = as.data.frame(X_train),
  y = y_train$Magnitude,
  method = "nnet",
  trControl = trainControl(method = 'cv', number = 5),
  metric = 'RMSE',  # Use RMSE for regression
  tuneGrid = expand.grid(size = c(5, 10, 15), decay = c(0.001, 0.01, 0.1))
)

# Predici i valori sulla base del set di test
predictions <- predict(grid_regressor, newdata = as.data.frame(X_test))

# Calcola l'errore quadratico medio (RMSE)
rmse <- sqrt(mean((predictions - y_test$Magnitude)^2))
cat("RMSE:", rmse, "\n")

