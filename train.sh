#!/bin/bash
{ python top100synth8metrics.py && python emballacc.py &&
aws s3 cp s3://camp-neuralnet-model-prod/prediction/top100-8metrics/ s3://camp-neuralnet-model-prod/prediction/top100-8metrics-backup/  --recursive &&
aws s3 cp ./models/a100-8metrics/ s3://camp-neuralnet-model-prod/prediction/top100-8metrics/  --recursive
} > ./log/train_log.txt
aws sns publish --topic-arn "arn:aws:sns:eu-west-1:147749137871:Train-Finish" --message file://log/train_log.txt