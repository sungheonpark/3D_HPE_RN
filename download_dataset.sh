#!/usr/bin/env bash

echo "Downloading training data"
wget -O data_train.tar https://www.dropbox.com/s/1cta5s1v49sqyui/data_train.tar?dl=1
echo "Downloading test data"
wget -O data_test.tar https://www.dropbox.com/s/qxztplajve33qkg/data_test.tar?dl=1
echo "Extracting training data"
tar xf data_train.tar
echo "Extracting test data"
tar xf data_test.tar
