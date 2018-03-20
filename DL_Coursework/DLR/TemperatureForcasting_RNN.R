library(keras)

# data prep
dir.create("~/Desktop/jena_climate", recursive=TRUE)
download.file(
    "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
    "~/Desktop/jena_climate/jena_climate_2009_2016.csv.zip"
)
unzip(
    "~/Desktop/jena_climate/jena_climate_2009_2016.csv.zip",
    exdir = "~/Desktop/jena_climate"
)

## inspecting library
library(tibble)
library(readr)
library(ggplot2)

data_dir <- "~/Desktop/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
glimpse(data)

## plotting sample
ggplot(data, aes(x=1:nrow(data), y = `T (degC)`)) + geom_line()

# data prep
data <- data.matrix(data[, -1])
train_data <- data[1:200000, ]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

# build generator
generator <- function(data, lookback, delay, min_index, max_index, 
                      shuffle = FALSE, batch_size = 128, step = 6) {
    if (is.null(max_index)) max_index <- nrow(data) - delay - 1
    i <- min_index + lookback
    function () {
        if (shuffle) {
            rows <- sample(c((min_index+lookback):max_index), size = batch_size)
        } else {
            if (i + batch_size >= max_index)
                i <<- min_index + lookback
            rows <- c(i:min(i+batch_size, max_index))
            i <<- i + length(rows)
        }
        
        samples <- array(0, dim = c(length(rows),
                                    lookback / step,
                                    dim(data)[[-1]]))
        
        targets <- array(0, dim = c(length(rows)))
        
        for (j in 1:length(rows)) {
            indices <- seq(rows[[j]] - lookback, rows[[j]],
                           length.out = dim(samples)[[2]])
            samples[j,,] <- data[indices,]
            targets[[j]] <- data[rows[[j]] + delay, 2]
        }
        list(samples, targets)
    }
}

# build model
lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 1,
    max_index = 200000,
    shuffle = TRUE,
    step = step,
    batch_size = batch_size
)

val_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 200001,
    max_index = 300000,
    step = step,
    batch_size = batch_size
)

test_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 300001,
    max_index = NULL,
    step = step,
    batch_size = batch_size
)

val_steps <- (300000 - 200001 - lookback) / batch_size
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

# establish a naive baseline

evaluate_naive_method <- function() {
    batch_maes <- c()
    for (step in 1:val_steps) {
        c(samples, targets) %<-% val_gen()
        preds <- samples[,dim(samples)[[2]],2]
        mae <- mean(abs(preds - targets))
        batch_maes <- c(batch_maes, mae)
    }
    print(mean(batch_maes))
}

baseline <- evaluate_naive_method()
celsius_baseline_mae <- baseline * std[[2]]

## basic model - flattens, so no recurrent/time series

model <- keras_model_sequential() %>%
    layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 10,
    validation_data = val_gen,
    validation_steps = val_steps
)

# Try Gated Recurrent Network - cheaper computationally than LSTM, less expressive power

model <- keras_model_sequential() %>%
    layer_gru(units = 32, input_shape = c(lookback / step, dim(data)[-1])) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 10,
    validation_data = val_gen,
    validation_steps = val_steps
)
