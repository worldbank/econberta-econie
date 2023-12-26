#!/bin/bash

export VERSION=$1
MODELS_FOLDER=$2
SUB_FOLDER=$3_"${VERSION}"

for FOLD in 0 1 2 3 4; do

  SERIALIZATION_DIR=${MODELS_FOLDER}/"${SUB_FOLDER}"/fold-${FOLD}

  python evaluate_allennlp_with_conll.py --model-path "${SERIALIZATION_DIR}" --fold "${FOLD}" --version "${VERSION}" --cuda-device -1 --batch-size 16

done

python cross_validation_entity_analysis.py analyze-cross-validation-errors --version "${VERSION}" --allennlp-model

python cross_validation_entity_analysis.py analyze-cross-validation-errors --version "${VERSION}" --allennlp-model --relations-only
