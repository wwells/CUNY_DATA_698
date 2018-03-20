# Workflow for Transferring ImageNet 

This workflow is to transfer the ImageNet tar files for research to Google Compute Object Storage so a CNN model can be trained and tested.  Researchers can apply for access to the tar files hosted by the Stanford Vision lab at http://image-net.org/download-images

Get VM with ~ 140 GiB Disk;   Make sure VM can get to outside world and has API perms to transfer to bucket. SSH to VM. 

Rough estimate is about 40 minutes to Pull, Untar, and synch per GiB.   

## 1. Pull Down Tar, check MD5

```
curl -o test.tar [URL TO]_test.tar \
	--retry 100 \
	--retry-delay 60 \
```

Compare MD5sum against those granted for research.

```
md5sum [filename.tar]
```

## 2a. Untar Archive (Validation / Test)

```
mkdir [validation_or_test]
tar -xf file_name.tar -C [validation_or_test]
```

## 2b. Untar Archive and Dirs Underneath (Training)

Special thanks to [Arun Das](https://github.com/arundasan91/Deep-Learning-with-Caffe/blob/master/Imagenet/How-to-properly-set-up-Imagenet-Dataset.md)

```
mkdir [train]
cd train/
```

Copy the following and paste it into a file (ex. tar_extract_script.sh).

```
#!/bin/bash
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xf ../$f)
done
rm -r *.tar
```

Give it executable permissions.

```
chmod 700 tar_extract_script.sh
```

Run the script.

```
./tar_extract_script.sh
```

## 3.  Rsynch to Google Storage

```
gsutil -m rsync . gs://image_net/[validation_or_test_or_train]
```
