library(keras)
#library(cloudml)

base_dir <- "~/Dropbox/CUNY/DATA-698_DeepLearning/Data/tiny-imagenet-200"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")

image_height <- 64
image_width <- 64
num_classes <- 200
num_training <- num_classes * 500
num_val <- num_classes * 50
batch_size <- 50
epochs <- 30
dropout1 <- 0.25
dropout2 <- 0.25
dropout3 <- 0.25
dropout4 <- 0.25

data_augmentation <- TRUE
modelname <- "localdev2"
modelsave <- paste0(modelname, ".h5")
historysave <- paste0(modelname, "_history.rds")
#logdir <- paste0(modelname, "_logs")
#dir.create(logdir)

## Model
model <- keras_model_sequential() %>%
    # Layer 1
    layer_conv_2d(filters = 32, kernel_size = c(6, 6), activation = "relu",
                  input_shape = c(image_height, image_width, 3)) %>%
    # Use max pooling and dropout
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(dropout1) %>%
    
    # Layer 2
    layer_conv_2d(filter = 32, kernel_size = c(3, 3), activation = "relu") %>%
    layer_conv_2d(filter = 64, kernel_size = c(3, 3), activation = "relu") %>%
    # Pooling and Dropout
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(dropout2) %>%
    
    # Layer 3
    layer_conv_2d(filter = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_conv_2d(filter = 128, kernel_size = c(3, 3), activation = "relu") %>%
    # Pooling and Dropout
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(dropout3) %>%
    
    # Classifier
    layer_flatten() %>%
    layer_dropout(dropout4) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = num_classes, activation = "softmax")

metric_top_5_categorical_accuracy <- function(y_true, y_pred) {
    metric_top_k_categorical_accuracy(y_true, y_pred, k = 5) 
}

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "categorical_crossentropy",
    metrics = c(top_5_categorical_accuracy = metric_top_5_categorical_accuracy)
)

## Flags to Add Data Augmentation
if (!data_augmentation) {
    train_datagen <- image_data_generator(rescale = 1/255)
} else {
    train_datagen <- image_data_generator(
        rescale = 1/255,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = TRUE
    )
}

validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
    train_dir,
    train_datagen,
    target_size = c(image_height, image_width),
    batch_size = batch_size,
    class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
    validation_dir,
    validation_datagen,
    target_size = c(image_height, image_width),
    batch_size = batch_size,
    class_mode = "categorical"
)

## callbacks and logs
#tensorboard(log_dir = logdir)

callbacks_list = list(
    callback_early_stopping(
        monitor = "top_5_categorical_accuracy",
        patience = 10
    ),
    callback_model_checkpoint(
        filepath = modelsave,
        monitor = "val_loss",
        save_best_only = TRUE
    ),
    callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.1,
        patience = 10
    )#,
    #callback_tensorboard(
    #    log_dir = "localdev1_logs",
    #    histogram_freq = 1
    #)
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = as.integer(num_training / batch_size),
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = as.integer(num_val / batch_size),
    callbacks = callbacks_list
)

saveRDS(history, historysave)
