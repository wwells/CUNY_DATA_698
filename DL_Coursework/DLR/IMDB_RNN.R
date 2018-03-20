library(keras)

max_features <- 10000
maxlen <- 500
batch_size <- 32

cat("loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences\n")

input_train <- pad_sequences(input_train, maxlen=maxlen)
input_test <- pad_sequences(input_test, maxlen=maxlen)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

# simple RNN model, with LSTM layer
model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 32) %>%
    layer_lstm(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop", 
    loss = "binary_crossentropy",
    metrics = c("acc")
)

history <- model %>% fit(
    input_train, 
    y_train,
    epochs = 10,
    batch_size = 128, 
    validation_split = 0.2
)