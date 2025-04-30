#!/bin/bash

DATASET=$1
DIR=$2
SEED=$3
SHOTS=$4

TRAINER=TrainerSupernet
CFG=fewshot
DATA="/data/zhenhan/datasets/pl_data"

if [ -d "${DIR}" ]; then
    echo "Directory '${DIR}' already exists, skip this job."
else
    CUDA_VISIBLE_DEVICES=0 python3 train_search.py \
        --dataset ${DATASET} \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
fi
