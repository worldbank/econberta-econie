#!/usr/bin/env bash

export VERSION='2.3'
export SLICE='1.0'
MODEL_FOLDER='../models/'
export N_ITER=30
export BATCH_SIZE=16
export EPOCHS=10
export GRADIENT_ACCUMULATION_STEPS=4
export DROPOUT=0.2
export DATASET_PATH=../data/econ_ie

for model_name in bert roberta mdeberta scratch pretrained; do

    if allennlp tune \
    allennlp/optuna_ner.jsonnet \
    allennlp/optuna-grid-search-hparams-small-${model_name}.json \
    --optuna-param-path allennlp/optuna-grid-search.json \
    --serialization-dir ${MODEL_FOLDER}/grid_search_ner_allennlp_"${VERSION}"_"${model_name}" \
    --study-name grid-search-causal-ner_"${VERSION}_${model_name}_${SLICE}" \
    --metrics test_f1-measure-overall \
    --direction maximize \
    --skip-if-exists \
    --n-trials $N_ITER; then

      echo "python3 load_best_allennlp_ner_models.py --version ${VERSION}_${model_name} --models-folder $MODEL_FOLDER --dataset_slice $SLICE"
      if python3 load_best_allennlp_ner_models.py --version ${VERSION}_${model_name} --models-folder $MODEL_FOLDER --dataset_slice $SLICE; then

        echo "sh evaluate_ner_allennlp_cv.sh ${VERSION}_${model_name} $MODEL_FOLDER ner_allennlp_best"
        sh evaluate_ner_allennlp_cv.sh ${VERSION}_${model_name} $MODEL_FOLDER ner_allennlp_best ${DATASET_PATH}

      fi

    fi

done
