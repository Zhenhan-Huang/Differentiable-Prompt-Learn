#!/bin/bash

WDIR=$1

for DATASET in Caltech101 DescribableTextures EuroSAT FGVCAircraft Food101 OxfordFlowers OxfordPets StanfordCars UCF101; do
    python3 parse_single.py \
        ${WDIR} \
        --nseeds 3 \
        --dataset ${DATASET} \
        --ci95
done
