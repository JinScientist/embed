# CAMP-ANN-training

## Summary
Prototype of neural network training service and prediction API

## Requirements

For training EC2 with GPU accelator, tensorflow,cuda kit,pandas are required. 

## Raw training data

Raw training data should be CSV file located in the 'csvdata' directory in the same loation as the programm. 

## Programm list

### Model for single account and multi metrics
* synth9metrics.py: training sample data formating and synthesization
* emb.py: training programm

### Model for multi accounts and multi metrics
* top100synth8metrics: training sample data formating and synthesization
* emballacc.py: training programm

### Prediction API prototype
* predfor100.py: load trained model file from S3 and run prediction


## Training service

    $ ./train.sh


