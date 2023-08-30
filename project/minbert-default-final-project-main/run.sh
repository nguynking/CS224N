#!/bin/bash
if [ "$1" = "pretrain" ]; then
    python3 classifier.py --option pretrain --epochs 5 --lr 1e-3 --hidden_dropout_prob 0.3 --batch_size 64 --use_gpu
elif [ "$1" = "finetune" ]; then
    python3 classifier.py --option finetune --epochs 5 --lr 1e-5 --hidden_dropout_prob 0.3 --batch_size 64 --use_gpu
else
    echo "Invalid Option Selected"
fi