#! /bin/bash

cd data
mkdir lossnet
mkdir train
cd train
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
rm train2014.zip
