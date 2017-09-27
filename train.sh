#!/bin/bash
python top100synth8metrics.py
python emballacc.py
aws s3 cp ./models/a100-8metrics/ s3://camp-neuralnet-model-prod/prediction/top100-8metrics/  --recursive
