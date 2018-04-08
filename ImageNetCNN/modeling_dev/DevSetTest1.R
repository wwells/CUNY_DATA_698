library(keras)
library(cloudml)

## devimagenet
img_size <- c(64, 64)
num_classes <- 200

num_training <- num_classes * 500
num_val <- num_classes * 50

batch_size <- 50

## Model
model <- keras_model_sequential() %>%
    ## CNN
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(64, 64, 3)) %>%
    layer_max_pooling_2d(pool_size = c(3, 3)) %>%
    ## Classifier
    layer_flatten() %>%
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

datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
    gs_local_dir("gs://tinyimage_net/train/"),
    datagen,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = "categorical"
)


history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = num_training / batch_size,
    epochs = 5#,
    #validation_data = validation_generator,
    #validation_steps = 50
)

