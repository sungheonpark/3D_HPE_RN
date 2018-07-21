#!/usr/bin/env bash

echo "Downloading training data"
wget -O data_train.tar https://www.dropbox.com/s/003y2dd8j1m70z8/data_train.tar?dl=1
echo "Downloading test data"
wget -O data_test.tar https://www.dropbox.com/s/dw0v3ur3f36rnf8/data_test.tar?dl=1
echo "Extracting training data"
tar xf data_train.tar
echo "Extracting test data"
tar xf data_test.tar
