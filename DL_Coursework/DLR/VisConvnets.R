library(keras)
library(cloudml)
#source("DogsVCats_DataPrep.R")

model <- load_model_hdf5('cats_and_dogs_small_1.h5')

img_path <- "/Users/wells_walter/Desktop/cats_and_dogs_small/test/cats/cat.1700.jpg"
img <- image_load(img_path, target_size = c(150,150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255
plot(as.raster(img_tensor[1,,,]))

#instantiate model from input tensor
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

activations <- activation_model %>% predict(img_tensor)

#first confolutional layer
first_layer_activation <- activations[[1]]

plot_channel <- function(channel) {
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(channel), axes = FALSE, asp = 1, 
          col = terrain.colors(12))
}

plot_channel(first_layer_activation[1,,,30])

## plot every channel in every intermediate activation

image_size <- 58
images_per_row <- 16

for (i in 1:8) {
    layer_activation <- activations[[i]]
    layer_name <- model$layers[[i]]$name
    
    n_features <- dim(layer_activation)[[4]]
    n_cols <- n_features %/% images_per_row
    
    png(paste0("cat_activations_", i, "_", layer_name, ".png"),
        width = image_size * images_per_row,
        height = image_size * n_cols)
    op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
    
    for (col in 0:(n_cols-1)) {
        for (row in 0:(images_per_row-1)) {
            channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
            plot_channel(channel_image)
        }
    }
    par(op)
    dev.off()
}
