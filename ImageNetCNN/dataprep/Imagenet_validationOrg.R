## The initial validation set in both TinyImageNet and the full Imagenet 
## are organized as series of images directly under the validation directory. 
## To use the great flow_images_from_directory() function in the Keras package, 
## we would prefer the validation images to be organized in subfolders representing 
## each class.   The work below helps organize the full ImageNet validation set
## into folders for each of the 1000 ImageNet Classes.   It relies on the synsets.csv 
## created using the dev kit and the validation ground truth file.

vals <- read.table('ILSVRC2012_validation_ground_truth.txt')
classes <- read.csv('synsets.csv')
images <- list.files('val_old')

vals$V2 <- classes$WNID[match(unlist(vals$V1), classes$ILSVRC2012.ID)]
vals$V3 <- images

classes <- as.character(unique(vals$V2))

#dir.create("validation")

for (class in classes) {
    newdir <- paste0("validation/", class)
    dir.create(newdir)
    
    temp <- vals[vals$V2 == class,]
    images <- as.character(temp$V3)
    
    for (image in images) {
        old_path <- paste0("val_old/", image)
        new_path <- paste0(newdir, "/", image)
        file.copy(old_path, new_path)
    }
}