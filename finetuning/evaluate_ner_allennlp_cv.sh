#!/bin/bash

export VERSION=$1
MODELS_FOLDER=$2
SUB_FOLDER=$3_"${VERSION}"
DATA_FOLDER=$4
CUDA_DEVICE=-1

for FOLD in 0 1 2 3 4; do

  SERIALIZATION_DIR=${MODELS_FOLDER}/"${SUB_FOLDER}"/fold-${FOLD}

  echo "python evaluate_allennlp_with_conll.py --model-path ${SERIALIZATION_DIR} --fold ${FOLD} --version ${VERSION} --dataset_folder ${DATA_FOLDER} --cuda-device ${CUDA_DEVICE} --batch-size 16"
  python evaluate_allennlp_with_conll.py --model-path "${SERIALIZATION_DIR}" --fold "${FOLD}" --version "${VERSION}" --dataset_folder "${DATA_FOLDER}" --cuda-device ${CUDA_DEVICE} --batch-size 16

done
