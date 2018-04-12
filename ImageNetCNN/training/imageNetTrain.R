library(keras)
library(cloudml)

## to submit to cloudml, mod file flag below to reflect environment

## Example of basic
# cloudml_train("imageNetTrain.R", master_type = "standard_gpu")

## Tuning HyperParameters
# cloudml_train("imageNetTrain.R", master_type = "standard_gpu", config = "tuning.yml")

## To change environment between local, dev, and full: 
# comment out or mod "file" in flags, update data augmentation flag as desired

## To review best runs in tensorboard check:
# tensorboard(ls_runs(order = metric_val_top_5_categorical_accuracy))[5,]

## Deploy and Predict
#cloudml_deploy("TinyImagenetModel", name = "TinyImageNetMod")
#cloudml_predict(list(as.vector(t(mnist_image))), name = "TinyImageNetMod",)

FLAGS <- flags(
    file = "dev_flags.yml",
    flag_string("train_dir", "gs://image_net/train"),
    flag_string("validation_dir", "gs://image_net/validation"),
    flag_string("model_name", "ImageNetModel"),
    flag_integer("image_height", 256), 
    flag_integer("image_width", 256), 
    flag_integer("num_classes", 1000),
    flag_integer("num_training", 1281167),
    flag_integer("num_val", 50000),
    flag_integer("batch_size", 20),
    flag_integer("epochs", 1000),
    
    ## tuned parameters
    flag_numeric("dropout_rate1", 0.75),
    flag_numeric("dropout_rate2", 0.75),
    flag_numeric("dropout_rate3", 0.75),
    flag_boolean("data_augmentation", TRUE)
)

print(FLAGS)

model <- keras_model_sequential() %>%
    # 4 layers
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), 
                  activation = "relu",
                  padding = "same",
                  input_shape = c(FLAGS$image_height, FLAGS$image_width, 3)) %>%
    layer_conv_2d(filters = 32, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_batch_normalization() %>%
    layer_dropout(FLAGS$dropout_rate1) %>%
    
    layer_conv_2d(filters = 32, 
                  kernel_size = c(3, 3), 
                  padding = "same",
                  activation = "relu") %>%
    layer_conv_2d(filters = 64, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_batch_normalization() %>%
    layer_dropout(FLAGS$dropout_rate2) %>%
    
    layer_conv_2d(filters = 128, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_batch_normalization() %>%
    
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1024, activation = "relu") %>%
    layer_dropout(rate = FLAGS$dropout_rate3) %>%
    layer_dense(units = FLAGS$num_classes, activation = "softmax")

## Defined metric, compile
metric_top_5_categorical_accuracy <- function(y_true, y_pred) {
    metric_top_k_categorical_accuracy(y_true, y_pred, k = 5) 
}

model %>% compile(
    optimizer = optimizer_adam(lr = 0.0001, decay = 1e-6),
    loss = "categorical_crossentropy",
    metrics = c('accuracy', top_5_accuracy = metric_top_5_categorical_accuracy)
)

## flags to add Data Augmentation
if (!FLAGS$data_augmentation) {
    train_datagen <- image_data_generator(rescale = 1/255)
} else {
    train_datagen <- image_data_generator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2, 
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = TRUE, 
        fill_mode = "nearest"
    )
}

validation_datagen <- image_data_generator(rescale = 1/255)

# create training and validation generators
train_generator <- flow_images_from_directory(
    gs_local_dir(FLAGS$train_dir),
    train_datagen,
    target_size = c(FLAGS$image_height, FLAGS$image_width),
    batch_size = FLAGS$batch_size,
    class_mode = "categorical", 
    shuffle = TRUE
)

validation_generator <- flow_images_from_directory(
    gs_local_dir(FLAGS$validation_dir),
    validation_datagen,
    target_size = c(FLAGS$image_height, FLAGS$image_width),
    batch_size = FLAGS$batch_size,
    class_mode = "categorical", 
    shuffle = TRUE
)

callbacks_list = list(
    callback_early_stopping(
        patience = 100
    )
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = as.integer(FLAGS$num_training / FLAGS$batch_size),
    epochs = FLAGS$epochs,
    validation_data = validation_generator,
    validation_steps = as.integer(FLAGS$num_val / FLAGS$batch_size),
    callbacks = callbacks_list
)

export_savedmodel(model, FLAGS$model_name)
