#!/usr/bin/env bash
# Accuracy should be 85% approx
python3 train_resnet.py $DATASETS/Food_100 $DATASETS/Food_100/trainlist.txt $DATASETS/Food_100/vallist.txt 100 -b 128 --ngpu 4 --schedule 40 80 --lr 0.2 --epochs 100 --attention_depth 0 --attention_width 0 --log_path ./logs/food_baseline