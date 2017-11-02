#!/bin/bash
python top100synth8metrics.py
python emballacc.py
aws sns publish --topic-arn "arn:aws:sns:eu-west-1:147749137871:Train-Finish" --message file://train_log.txt
aws s3 cp ./models/a100-8metrics/ s3://camp-neuralnet-model-prod/prediction/top100-8metrics/  --recursive
