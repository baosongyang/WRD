#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

output=word_order.y
task=word_order

python thumt/bin/word_order.py  \
    --models transformer_ori \
    --eval_steps 10 \
    --softmax_size 1000 \
    --input \
        ./TrainData/$task.txt.tr.x \
        ./TrainData/$task.txt.tr.y \
    --eval \
        ./TrainData/$task.txt.va.x \
        ./TrainData/$task.txt.va.y \
    --test \
        ./TrainData/$task.txt.te.x \
        ./TrainData/$task.txt.te.y \
    --output \
        out/trans/$output \
    --vocabulary \
        ./TrainData/vocab.en.txt \
        ./TrainData/vocab.en.txt \
    --checkpoints \
        eval \
    --parameters \
        "device_list=[0],decode_batch_size=500,num_encoder_layers=6" \
