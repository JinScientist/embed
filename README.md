# CAMP-ANN-training

## Summary
Prototype of neural network training service and prediction API for usage time series prediction

## Requirements

For training EC2 with GPU accelator, Tensorflow,Cuda kit,Pandas are required. 

## Raw training data

Raw feature data should be CSV file located in the 'csvdata' directory in the same loation as the programm. SQL script for aggregating and pre-formatting raw feature data from Athena is in 'query_for_training.sql'

## Programm list

### Model for single account and multi metrics
* synth9metrics.py: training sample data formating and synthesization
* emb.py: training programm

### Model for multi accounts and multi metrics
* top100synth8metrics: training sample data formating and synthesization
* emballacc.py: training programm

### Prediction API prototype
* predfor100.py: load trained model file from S3 and run prediction


## Train model and upload to S3 for multi accounts with multi metric
    
    $ sudo chmod +x ./train.sh
    $ ./train.sh > ./trainemblog.txt


