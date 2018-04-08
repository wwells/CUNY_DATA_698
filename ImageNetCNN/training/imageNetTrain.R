library(keras)
library(cloudml)

## to submit to cloudml, mod file flag below to reflect environment

## Example of basic
# cloudml_train("imageNetTrain.R", master_type = "standard_gpu")

## Tuning HyperParameters
# cloudml_train("imageNetTrain.R", master_type = "standard_gpu", config = "tuning.yml")

## To run over full dataset:
# comment out "file" in flags, update data augmentation flag as desired

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
    flag_integer("batch_size", 100),
    flag_integer("epochs", 1000),
    
    ## tuned paramaters
    flag_numeric("learning_rate", 0.01),
    flag_numeric("dropout_rate", 0.1),
    flag_integer("initial_kernel_size", 3),
    flag_boolean("data_augmentation", FALSE),
    flag_integer("filter_layer1", 32),
    flag_integer("filter_layer2", 64),
    flag_integer("filter_layer3", 128),
    flag_integer("filter_layer4", 128),
    flag_integer("filter_dense", 512)
)

print(FLAGS)

model <- keras_model_sequential() %>%
    # 4 layers
    layer_conv_2d(filters = FLAGS$filter_layer1, 
                  kernel_size = c(FLAGS$initial_kernel_size, FLAGS$initial_kernel_size), 
                  activation = "relu",
                  input_shape = c(FLAGS$image_height, FLAGS$image_width, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = FLAGS$filter_layer2, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = FLAGS$filter_layer3, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = FLAGS$filter_layer4, 
                  kernel_size = c(3, 3), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    
    # Classifier
    layer_flatten() %>%
    layer_dropout(FLAGS$dropout_rate) %>%
    layer_dense(units = FLAGS$filter_dense, activation = "relu") %>%
    layer_dense(units = FLAGS$num_classes, activation = "softmax")

## Defined metric, compile
metric_top_5_categorical_accuracy <- function(y_true, y_pred) {
    metric_top_k_categorical_accuracy(y_true, y_pred, k = 5) 
}

model %>% compile(
    optimizer = optimizer_rmsprop(lr = FLAGS$learning_rate),
    loss = "categorical_crossentropy",
    metrics = c('accuracy', top_5_categorical_accuracy = metric_top_5_categorical_accuracy)
)

## flags to add Data Augmentation
if (!FLAGS$data_augmentation) {
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

# create training and validation generators
train_generator <- flow_images_from_directory(
    gs_local_dir(FLAGS$train_dir),
    train_datagen,
    target_size = c(FLAGS$image_height, FLAGS$image_width),
    batch_size = FLAGS$batch_size,
    class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
    gs_local_dir(FLAGS$validation_dir),
    validation_datagen,
    target_size = c(FLAGS$image_height, FLAGS$image_width),
    batch_size = FLAGS$batch_size,
    class_mode = "categorical"
)


callbacks_list = list(
    callback_early_stopping(
        monitor = "top_5_categorical_accuracy",
        patience = 20
    ),
    callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.005,
        patience = 20
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
