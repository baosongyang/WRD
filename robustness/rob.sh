#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

output=word_order.y
task=word_order

python ../thumt/bin/translator.py --models transformer_ori --input newstest2014.bpe.en.recover --output $task.bpe.recover.ori --vocabulary ../TrainData/vocab.en.txt ../TrainData/vocab.de.txt  --checkpoints ../eval_ori --parameters=device_list=[1]

python ../thumt/bin/translator.py --models transformer_di --input newstest2014.bpe.en.recover --output $task.bpe.recover.di --vocabulary ../TrainData/vocab.en.txt ../TrainData/vocab.de.txt  --checkpoints ../eval_bidi --parameters=device_list=[1]

python ../thumt/bin/translator.py --models rnnp --input newstest2014.bpe.en.recover --output $task.bpe.recover.rnn --vocabulary ../TrainData/vocab.en.txt ../TrainData/vocab.de.txt  --checkpoints ../eval_rnn --parameters=device_list=[1]


python ../thumt/bin/translator.py --models transformer_ori --input newstest2014.bpe.en.reorder --output $task.bpe.reorder.ori --vocabulary ../TrainData/vocab.en.txt ../TrainData/vocab.de.txt  --checkpoints ../eval_ori --parameters=device_list=[1]

python ../thumt/bin/translator.py --models transformer_di --input newstest2014.bpe.en.reorder --output $task.bpe.reorder.di --vocabulary ../TrainData/vocab.en.txt ../TrainData/vocab.de.txt  --checkpoints ../eval_bidi --parameters=device_list=[1]

python ../thumt/bin/translator.py --models rnnp --input newstest2014.bpe.en.reorder --output $task.bpe.reorder.rnn --vocabulary ../TrainData/vocab.en.txt ../TrainData/vocab.de.txt  --checkpoints ../eval_rnn --parameters=device_list=[1]
