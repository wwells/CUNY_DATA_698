## The initial validation set in both TinyImageNet and the full Imagenet 
## are organized as series of images directly under the validation directory. 
## To use the great flow_images_from_directory() function in the Keras package, 
## we would prefer the validation images to be organized in subfolders representing 
## each class.   The work below helps organize the tinyImageNet validation set
## into folders for each of the 200 TinyImageNet Classes.

vals <- read.table('val/val_annotations.txt')
images <- as.list(as.character(vals$V1))
labels <- as.list(as.character(vals$V2))

classes <- as.character(unique(vals$V2))

dir.create("validation")

for (class in classes) {
    newdir <- paste0("validation/", class)
    dir.create(newdir)
    
    temp <- vals[vals$V2 == class,]
    images <- as.character(temp$V1)
    
    for (image in images) {
        old_path <- paste0("val/images/", image)
        new_path <- paste0(newdir, "/", image)
        file.copy(old_path, new_path)
    }
}

