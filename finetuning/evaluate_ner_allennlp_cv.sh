#!/bin/bash

export VERSION=$1
MODELS_FOLDER=$2
SUB_FOLDER=$3_"${VERSION}"
DATA_FOLDER=../data/econ_ie

for FOLD in 0 1 2 3 4; do

  SERIALIZATION_DIR=${MODELS_FOLDER}/"${SUB_FOLDER}"/fold-${FOLD}

  echo "python evaluate_allennlp_with_conll.py --model-path ${SERIALIZATION_DIR} --fold ${FOLD} --version ${VERSION} --dataset_folder ${DATA_FOLDER} --cuda-device -1 --batch-size 16"
  python evaluate_allennlp_with_conll.py --model-path "${SERIALIZATION_DIR}" --fold "${FOLD}" --version "${VERSION}" --dataset_folder ${DATA_FOLDER} --cuda-device -1 --batch-size 16

done
