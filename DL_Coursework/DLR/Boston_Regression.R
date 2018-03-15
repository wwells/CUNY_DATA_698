library(keras)

boston <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston

#normalize data
mean <- apply(train_data, 2, mean)
std <- apply (train_data, 2, sd)
train_data <- scale(train_data, center=mean, scale=std)
test_data <- scale(test_data, center=mean, scale=std)

# build network
build_model <- function() {
    model <- keras_model_sequential() %>%
        layer_dense(units = 64, activation = "relu", 
                    input_shape = dim(train_data)[[2]]) %>%
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 1)
    
    model %>% compile(
        optimizer = "rmsprop",
        loss = "mse", 
        metrics = c("mae")
    )
}

# cross validation
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 100
all_scores <- c()
all_mae_histories <- NULL
for (i in 1:k) {
    cat("processing fold #", i, "\n")
    
    val_indices <- which(folds == i, arr.ind=TRUE)
    
    val_data <- train_data[val_indices,]
    val_targets <- train_data[val_indices]
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    model <- build_model()
    history <- model %>% fit(
        partial_train_data,
        partial_train_targets,
        validation_data = list(val_data, val_targets),
        epochs = num_epochs,
        batch_size = 1, 
        verbose = 0
    )
    
    mae_history <- history$metrics$val_mean_absolute_error
    all_mae_histories <- rbind(all_mae_histories, mae_history)
    results <- model %>% evaluate(val_data, val_targets, verbose = 0)
    all_scores <- c(all_scores, results$mean_absolute_error)
}