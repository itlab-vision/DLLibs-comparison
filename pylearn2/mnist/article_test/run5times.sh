#!/bin/bash

cd article_cnn

for i in 1 2 3 4 5
do
    python exec_cnn_train.py
    cd ../article_test
    python exec_mlp_train.py
    cd ../article_cnn
done
