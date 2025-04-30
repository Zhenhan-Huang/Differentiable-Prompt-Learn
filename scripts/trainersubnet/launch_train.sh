#!/bin/bash

DATASET=$1
DIR=$2
IMG_ALPHA_PATH=$3
TXT_ALPHA_PATH=$4
SEED=$5
SHOTS=$6

TRAINER=TrainerSubnet
CFG=fewshot
DATA="/path/to/dataset/folder"

if [ -d "${DIR}" ]; then
    echo "Directory '${DIR}' already exists, skip this job."
else
    CUDA_VISIBLE_DEVICES=0 python3 train.py \
        --dataset ${DATASET} \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.TRAINERSUBNET.IMG_ALPHA_PATH $IMG_ALPHA_PATH \
        TRAINER.TRAINERSUBNET.TXT_ALPHA_PATH $TXT_ALPHA_PATH \
        DATASET.SUBSAMPLE_CLASSES all
fi