#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
from typing import List, Union, Dict
from tqdm import tqdm
from utils import get_or_create_path

import fire


class Sentence:

    def __init__(self, document_id: str, index: int, tokens: List[str],
                 gold_labels: List[str], predicted_labels: List[str], fold: int):
        self._id = f'{document_id}_{index}'
        self._document_id = document_id
        self._index = index
        self._tokens = tokens
        self._text = ' '.join(tokens)
        self._gold_labels = gold_labels
        self._gold_classes = [get_category(label) for label in gold_labels]
        self._predicted_labels = predicted_labels
        self._predicted_classes = [get_category(label) for label in predicted_labels]
        self._prediction_error_classes = [{'gold': gold_class, 'predicted': predicted_class} for
                                          predicted_class, gold_class in zip(self.predicted_classes, self.gold_classes)
                                          if gold_class != predicted_class]
        self._fold = fold

    @property
    def id(self) -> str:
        return self._id

    @property
    def document_id(self) -> str:
        return self._document_id

    def index(self) -> int:
        return self._index

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    @property
    def text(self) -> str:
        return self._text

    @property
    def gold_labels(self) -> List[str]:
        return self._gold_labels

    @property
    def gold_classes(self) -> List[str]:
        return self._gold_classes

    @property
    def predicted_labels(self) -> List[str]:
        return self._predicted_labels

    @property
    def predicted_classes(self) -> List[str]:
        return self._predicted_classes

    @property
    def fold(self) -> int:
        return self._fold

    @property
    def prediction_errors_classes(self) -> List[Dict[str, str]]:
        return self._prediction_error_classes

    def contains_prediction_errors(self) -> bool:
        return len(self.prediction_errors_classes) > 0

    def contains_prediction_errors_for_class(self, entity_class: str, unlabelled_only: bool = False):
        if unlabelled_only:
            if entity_class:
                return len([error for error in self.prediction_errors_classes if
                            error['gold'] == 'O' and error['predicted'] == entity_class]) > 0
            else:
                return len([error for error in self.prediction_errors_classes if
                            error['gold'] == 'O' and error['predicted'] != 'O']) > 0
        else:
            return entity_class in [error['gold'] for error in self.prediction_errors_classes] + \
                [error['predicted'] for error in self.prediction_errors_classes if
                 error['gold'] != 'O' and error['predicted'] == entity_class]

    def get_annotation_errors(self):
        _tokens = []
        for idx, token in enumerate(self.tokens):
            if self.gold_classes[idx] != self.predicted_classes[idx]:
                _tokens.append(get_annotation_error(token, self.gold_labels[idx], self.predicted_labels[idx]))
            else:
                _tokens.append(token)
        return ' '.join(_tokens)

    def to_json(self):
        return {
            'Document': self.document_id,
            'Index': self.index,
            'Fold': self.fold,
            'Text': self.get_annotation_errors()
        }

    def __iter__(self):
        for token, predicted_class, predicted_label, gold_class, gold_label in \
                zip(self.tokens, self.predicted_classes, self.predicted_labels, self.gold_classes, self.gold_labels):
            yield token, predicted_class, predicted_label, gold_class, gold_label

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return self.text


def get_category(label) -> str:
    return label.split('-')[1] if '-' in label else label


def get_annotation_error(token, gold_label, predicted_label) -> str:
    return f'{token}  ({predicted_label},{gold_label})'


def load_sentences_and_predictions(conll_test_file_path: Path, predictions_file_path: Path, fold: int = None) \
        -> List[Sentence]:
    sentences = []
    tokens = []
    gold_labels = []
    predicted_labels = []
    document = f'{conll_test_file_path.stem}-{fold}{conll_test_file_path.suffix}'
    index = 0
    for test_line, prediction_line in zip(conll_test_file_path.open(mode='r', encoding='utf8').readlines(),
                                          predictions_file_path.open(mode='r', encoding='utf8').readlines()):
        if test_line.strip() == '':
            if len(tokens) > 0:
                sentences.append(
                    Sentence(document_id=document, index=index, tokens=tokens, gold_labels=gold_labels,
                             predicted_labels=predicted_labels, fold=fold)
                )
                tokens = []
                gold_labels = []
                predicted_labels = []
                index += 1
        else:
            test_parts = test_line.split()
            prediction_parts = prediction_line.split()
            assert test_parts[0] == prediction_parts[0]
            tokens.append(test_parts[0])
            gold_labels.append(prediction_parts[-2])
            predicted_labels.append(prediction_parts[-1])

    return sentences


def convert_allennlp_predictions_to_conll(allennlp_predictions_path: Union[str, Path],
                                          conll_input_path: Union[str, Path], fold: int, version: str,
                                          dataset_type: str) -> Path:
    sentences = load_sentences_and_predictions(conll_test_file_path=Path(conll_input_path),
                                               predictions_file_path=allennlp_predictions_path, fold=fold)
    if fold is not None:
        predictions_path = get_or_create_path(f'predictions/version_{version}/fold-{fold}')
    else:
        predictions_path = get_or_create_path(f'predictions/version_{version}')

    predictions_file_name = f'predictions_allennlp_merged_{dataset_type}.txt'
    predictions_path = predictions_path / predictions_file_name

    with predictions_path.open(mode='w', encoding='utf8') as out_file:
        for sentence in tqdm(sentences, 'Writing predictions...'):
            for token_label in sentence:
                predicted_label, gold_label = token_label[4], token_label[2]
                out_file.write(
                    ' '.join([token_label[0], sentence.document_id, predicted_label, gold_label]) + '\n'
                )
            out_file.write('\n')

    return predictions_path


if __name__ == '__main__':
    fire.Fire()
