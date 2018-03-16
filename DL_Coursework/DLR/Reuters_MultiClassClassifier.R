library(keras)

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

# prep data
vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- matrix(0, nrow = length(sequences), ncol=dimension)
    for (i in 1:length(sequences))
        results[i, sequences[[i]]] <- 1
    results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# one hot encoding (manual function)
to_one_hot <- function(labels, dimension = 46) {
    results <- matrix(0, nrow = length(labels), ncol = dimension)
    for (i in 1:length(labels))
        results[i, labels[[i]] + 1] <- 1
    results
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)
#using keras can also be done with `to_categorical(train_labels)`

# setup validation set
val_indices <- 1:1000
x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices,]
y_val <- one_hot_train_labels[val_indices,]
partial_y_train <- one_hot_train_labels[-val_indices,]


# setup NN architecture
model <- keras_model_sequential() %>%
    layer_dense(units = 128, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

# compile model
model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

# train model
history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 15,
    batch_size = 512, 
    validation_data = list(x_val, y_val)
)

results <- model %>% evaluate(x_test, one_hot_test_labels)
predictions <- model %>% predict(x_test)