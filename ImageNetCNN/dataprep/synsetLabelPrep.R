## convert the matlab file provided in the development kit to R df class Ids and labels
## Researchers can apply for access to the tar files hosted by the Stanford Vision lab at http://image-net.org/download-images

if (!require('r.Matlab')) (install.packages('r.Matlab'))

df <- readMat('data/meta.mat')
df <- data.frame(t(as.data.frame(df$synsets)))
df <- dplyr::select(df, "ILSVRC2012.ID", "WNID", "words", "gloss", "num.train.images")
names <- names(df)

l <- unlist(df)
tf <- matrix(l, 1860, 5)
df <- data.frame(tf)
names(df) <- names

write.csv(df, "data/synsets.csv")