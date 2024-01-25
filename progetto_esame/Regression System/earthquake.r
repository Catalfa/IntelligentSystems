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

# Carica le librerie necessarie
library(caret)
library(dplyr)
library(maps)
library(mapdata)
library(mapproj)

# Carica il file CSV
Data <- read.csv("progetto_esame/dataset/database.csv")

# Specifica la proporzione per la divisione (ad esempio, 0.8 per l'80% dei dati nel primo file)
split_ratio <- 0.8

# Calcola il numero di righe per ciascun file
num_rows <- nrow(Data)
num_rows_first <- round(num_rows * split_ratio)

# Suddividi il dataframe in due parti
train_orig <- Data[1:num_rows_first, ]
test_orig <- Data[(num_rows_first + 1):num_rows, ]

data <- train_orig[, c('Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude')]

# Creazione della colonna Timestamp
timestamp <- vector("numeric", length = nrow(data))
for (i in 1:nrow(data)) {
  tryCatch({
    ts <- as.POSIXct(strptime(paste(data$Date[i], data$Time[i]), format="%m/%d/%Y %H:%M:%S"))
    timestamp[i] <- as.numeric(ts)
  }, error = function(e) {
    # print('ValueError')
    timestamp[i] <- NA
  })
}

# Aggiunge la colonna Timestamp al dataframe
data$Timestamp <- timestamp

# Rimuovi le righe con 'ValueError'
final_data <- data[!is.na(data$Timestamp) & data$Timestamp != 'ValueError', ]

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

# Crea il grafico
plot(xy, col = 'blue', pch = 16, cex = 0.5, main = "All affected areas")
map("world", add = TRUE, col = "black", fill = FALSE)