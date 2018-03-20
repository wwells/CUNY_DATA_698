library(keras)
library(cloudml)

base_dir <- "gs://dogs_v_cats"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    loss = "binary_crossentropy", 
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
)

## prep images
# data augmentation
datagen <- image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2, 
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE, 
    fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
    gs_local_dir(train_dir),
    datagen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
    gs_local_dir(validation_dir),
    test_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50
)

export_savedmodel(model, "cats_and_dogs_small2")
model %>% save_model_hdf5("cats_and_dogs_small_2.h5")