#!/usr/bin/env bash

VERSION='2.3'
SLICE='1.0'
MODEL_FOLDER='../models/'
N_ITER=30
BATCH_SIZE=16
EPOCHS=10
gradient_accumulation_steps=4
dropout=0.2

for model_name in bert roberta mdeberta scratch pretrained
do
    allennlp tune \
    allennlp/optuna_ner.jsonnet \
    allennlp/optuna-grid-search-hparams-small-${model_name}.json \
    --optuna-param-path allennlp/optuna-grid-search.json \
    --serialization-dir ${MODEL_FOLDER}/grid_search_ner_allennlp_"${VERSION}" \
    --study-name grid-search-causal-ner_"${VERSION}_${model_name}_${SLICE}" \
    --metrics test_f1-measure-overall \
    --direction maximize \
    --skip-if-exists \
    --n-trials $N_ITER 
    
    python3 load_best_allennlp_ner_models.py --version ${VERSION}_${model_name} --models-folder $MODEL_FOLDER --dataset_slice $SLICE
    
    sh evaluate_ner_allennlp_cv.sh ${VERSION}_${model_name}_${SLICE} $MODEL_FOLDER ner_allennlp_best
done
