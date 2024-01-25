#Importing Library
#install.packages("caret")
#install.packages("randomForest")
library(caret)

#Importing Training & Testing Dataset
train_orig <- read.csv("progetto_esame/dataset/Mnist_Train.csv")
test_orig <- read.csv("progetto_esame/dataset/Mnist_Test.csv")

ncol_train <- ncol(train_orig)
ncol_test <- ncol(test_orig)

# Importa la libreria
library(randomForest)

# Estrai le etichette dal dataframe di addestramento
train_orig_labels <- train_orig[, 1]

# Converte le etichette in un fattore
train_orig_labels <- as.factor(train_orig_labels)

# Rimuovi la colonna "label" dai dati di addestramento
train_data <- train_orig[, -1]

# Numero di alberi da costruire
num_trees <- 100

# Addestra il modello random forest
rf <- randomForest(x = train_data, y = train_orig_labels, ntree = num_trees)

# Visualizza i risultati del modello
print(rf)

# output predictions for submission
test_data <- test_orig[, -1]
predictions <- data.frame(ImageId = seq_len(nrow(test_orig)),
                           Label = levels(train_orig_labels)[predict(rf, test_data)])

head(predictions)

write.csv(predictions, "rf_benchmark.csv")
