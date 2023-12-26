#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from utils import get_or_create_path
from collections import defaultdict

import pandas as pd
import fire


class Sentence:

    def __init__(self, document_id: str, line_index: int, sentence_index: int, tokens: List[str],
                 gold_labels: List[str], predicted_labels: List[str], fold: int, relation_flags: List[bool]):
        self._id = f'{document_id}_{str(line_index)}-{str(sentence_index)}'
        self._document_id = document_id
        self._line_index = line_index
        self._sentence_index = sentence_index
        self._tokens = tokens
        self._text = ' '.join(tokens)
        self._gold_labels = gold_labels
        self._gold_classes = [get_category(label) for label in gold_labels]
        self._predicted_labels = predicted_labels
        self._predicted_classes = [get_category(label) for label in predicted_labels]
        self._prediction_error_classes = [{'gold': gold_class, 'predicted': predicted_class} for
                                          predicted_class, gold_class in zip(self.predicted_classes, self.gold_classes)
                                          if gold_class != predicted_class]
        self._relation_flags = relation_flags
        self._fold = fold
        self._sector = document_id.split('_')[1]

    @property
    def id(self) -> str:
        return self._id

    @property
    def document_id(self) -> str:
        return self._document_id

    @property
    def line_index(self) -> int:
        return self._line_index

    @property
    def sentence_index(self) -> int:
        return self._sentence_index

    @property
    def sector(self) -> str:
        return self._sector

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
    def relation_flags(self) -> List[bool]:
        return self._relation_flags

    @property
    def prediction_errors_classes(self) -> List[str]:
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
            'Sector': self.sector,
            'Line': self.line_index,
            'Sentence': self.sentence_index,
            'Fold': self.fold,
            'Text': self.get_annotation_errors()
        }

    def __iter__(self):
        for token, predicted_class, predicted_label, gold_class, gold_label, relation_flag in \
                zip(self.tokens, self.predicted_classes, self.predicted_labels, self.gold_classes, self.gold_labels,
                    self.relation_flags):
            yield token, predicted_class, predicted_label, gold_class, gold_label, relation_flag

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return self.text


def load_sentences(conll_test_file_path: Path, fold: int = None, predictions: List[List[str]] = None) \
        -> Tuple[List[Sentence], List[List[str]]]:
    sentences = []
    tokens = []
    relation_flags = []
    gold_labels = []
    predicted_labels = []
    document = None
    line_index, sentence_index = None, None
    sentence_count = 0
    error_count = 0
    skip_indices = []
    for line in conll_test_file_path.open(mode='r', encoding='utf8').readlines():
        if line.strip() == '':
            if len(tokens) > 0:
                try:
                    if predictions is not None:
                        assert len(predicted_labels) == 0
                        assert len(tokens) == len(predictions[sentence_count])
                        predicted_labels = predictions[sentence_count]
                    sentences.append(
                        Sentence(document_id=document, line_index=line_index, sentence_index=sentence_index,
                                 tokens=tokens, gold_labels=gold_labels, predicted_labels=predicted_labels,
                                 relation_flags=relation_flags, fold=fold)
                    )
                except AssertionError:
                    error_count += 1
                    skip_indices.append(sentence_count)
                tokens = []
                relation_flags = []
                gold_labels = []
                predicted_labels = []
                sentence_count += 1
        else:
            parts = line.split()
            tokens.append(parts[0])
            if len(parts) == 3:
                assert predictions is None
                gold_labels.append(parts[1])
                predicted_labels.append(parts[2])
            elif len(parts) == 4:
                document = parts[1]
                if '_' in parts[2]:
                    line_sentence_index, relation_flag = parts[2].split('_')
                    line_sentence_index = line_sentence_index.replace('B-', '')
                    relation_flag = relation_flag.replace('R-', '') == 'True'
                else:
                    line_sentence_index, relation_flag = parts[2], None
                    line_sentence_index = line_sentence_index.replace('B-', '')
                line_index, sentence_index = line_sentence_index.split('-')
                gold_labels.append(parts[3])
                relation_flags.append(relation_flag)
                assert predictions is not None
            elif len(parts) == 5:
                assert predictions is None
                document = parts[1]
                if '_' in parts[2]:
                    line_sentence_index, relation_flag = parts[2].split('_')
                    line_sentence_index = line_sentence_index.replace('B-', '')
                    relation_flag = relation_flag.replace('R-', '') == 'True'
                else:
                    line_sentence_index, relation_flag = parts[2], None
                    line_sentence_index = line_sentence_index.replace('B-', '')
                line_index, sentence_index = [int(part) for part in line_sentence_index.split('-')]
                relation_flags.append(relation_flag)
                gold_labels.append(parts[3])
                predicted_labels.append(parts[4])

    if predictions:
        predictions = [prediction for idx, prediction in enumerate(predictions) if idx not in skip_indices]
    return sentences, predictions


def load_sentences_and_predictions(conll_test_file_path: Path, predictions_file_path: Path, fold: int = None) \
        -> List[Sentence]:
    sentences = []
    tokens = []
    relation_flags = []
    gold_labels = []
    predicted_labels = []
    document = None
    line_index, sentence_index = None, None
    sentence_count = 0
    for test_line, prediction_line in zip(conll_test_file_path.open(mode='r', encoding='utf8').readlines(),
                                          predictions_file_path.open(mode='r', encoding='utf8').readlines()):
        if test_line.strip() == '':
            if len(tokens) > 0:
                sentences.append(
                    Sentence(document_id=document, line_index=line_index, sentence_index=sentence_index,
                             tokens=tokens, gold_labels=gold_labels, predicted_labels=predicted_labels,
                             relation_flags=relation_flags, fold=fold)
                )
                tokens = []
                relation_flags = []
                gold_labels = []
                predicted_labels = []
                sentence_count += 1
        else:
            test_parts = test_line.split()
            prediction_parts = prediction_line.split()
            assert test_parts[0] == prediction_parts[0]
            tokens.append(test_parts[0])
            document = test_parts[1]
            line_sentence_index, relation_flag = test_parts[2].split('_')
            line_sentence_index = line_sentence_index.replace('B-', '')
            relation_flag = relation_flag.replace('R-', '') == 'True'
            line_index, sentence_index = [int(part) for part in line_sentence_index.split('-')]
            gold_labels.append(prediction_parts[-2])
            predicted_labels.append(prediction_parts[-1])
            relation_flags.append(relation_flag)

    return sentences


def load_sentences_for_fold(predictions_path: Path, fold: int, discriminator_suffix: str, allennlp_model: bool = False,
                            relations_only: bool = False) -> Dict[str, List[Sentence]]:
    sentences_per_dataset = dict()

    for dataset_type in ['dev', 'test']:

        if relations_only:
            if allennlp_model:
                predictions_file_name = f'predictions_allennlp_merged_{dataset_type}{discriminator_suffix}' \
                                        f'_relations.txt'
            else:
                predictions_file_name = f'predictions_{dataset_type}{discriminator_suffix}_relations.txt'
        else:
            if allennlp_model:
                predictions_file_name = f'predictions_allennlp_merged_{dataset_type}{discriminator_suffix}.txt'
            else:
                predictions_file_name = f'predictions_{dataset_type}{discriminator_suffix}.txt'

        fold_predictions = predictions_path / f'fold-{fold}' / predictions_file_name
        sentences, _ = load_sentences(fold_predictions, fold=fold)
        sentences_per_dataset[dataset_type] = sentences

    return sentences_per_dataset


def load_sentences_per_fold(predictions_per_fold_path: Path, number_of_folds: int = 5, allennlp_model: bool = False,
                            relations_only: bool = False, discriminator: str = None) \
        -> Dict[int, Dict[str, List[Sentence]]]:
    discriminator_suffix = f'_{discriminator}' if discriminator else ''
    sentences_per_fold = defaultdict(dict)

    for fold in range(number_of_folds):
        sentences_per_fold[fold] = load_sentences_for_fold(predictions_path=predictions_per_fold_path,
                                                           fold=fold,
                                                           discriminator_suffix=discriminator_suffix,
                                                           allennlp_model=allennlp_model,
                                                           relations_only=relations_only)

    return sentences_per_fold


red = '\033[91m'
green = '\033[92m'
end = '\033[0m'


def get_category(label) -> str:
    return label.split('-')[1] if '-' in label else label


def print_for_notebook(token, gold_label, predicted_label):
    print(red + token + end, end=' ')
    print('(' + red + predicted_label + end + ',' + green + gold_label + end + ')', end=' ')


def get_annotation_error(token, gold_label, predicted_label):
    return token + ' ' + '(' + predicted_label + ',' + gold_label + ')'


def print_for_latex(token, real_label, predicted_label):
    print('\\textcolor{red}{%s} (\\textcolor{red}{%s},\\textcolor{green}{%s})' % (token, predicted_label, real_label))


def parse_results(sentences_per_fold: Dict[int, Dict[str, List[Sentence]]], for_latex: bool = False,
                  filter_category: str = None, unlabelled_only: bool = False) -> List[Sentence]:
    sentences_with_error = []
    total_sentences = 0
    for fold in sentences_per_fold.keys():
        for dataset_type in sentences_per_fold[fold]:
            sentences = sentences_per_fold[fold][dataset_type]
            total_sentences += len(sentences)
            print('Analyzing %s sentences from fold %s' % (len(sentences), fold))
            for sentence in sentences:
                if sentence.contains_prediction_errors():
                    if unlabelled_only and sentence.contains_prediction_errors_for_class(filter_category,
                                                                                         unlabelled_only):
                        sentences_with_error.append(sentence)
                    elif not unlabelled_only and filter_category is None:
                        sentences_with_error.append(sentence)
    print('')
    print('Found %s sentences with errors\n' % len(sentences_with_error))

    sentences_with_error.sort(
        key=lambda _sentence: (_sentence.document_id, _sentence.line_index, _sentence.sentence_index)
    )

    errors_per_sector = defaultdict(list)
    for sentence in sentences_with_error:
        errors_per_sector[sentence.sector].append(sentence)

    for sector in errors_per_sector:
        print(f'------------------------------{sector.upper()}------------------------------\n')
        print(f'Printing errors for sector {sector}\n\n')
        for sentence in errors_per_sector[sector]:
            if filter_category is None or sentence.contains_prediction_errors_for_class(filter_category,
                                                                                        unlabelled_only):
                print('Line %s, sentence %s from document %s / fold %s: ' % (
                    sentence.line_index, sentence.sentence_index, sentence.document_id, sentence.fold
                ))
                for token, predicted_class, predicted_label, gold_class, gold_label, relation_flag in sentence:
                    if predicted_class != gold_class:
                        if for_latex:
                            print_for_latex(token, gold_label, predicted_label)
                        else:
                            print_for_notebook(token, gold_label, predicted_label)
                    else:
                        print(token, end=' ')
                print('\n')
        print('\n\n----------------------------------------------------------------------------\n\n')

    return sentences_with_error


def merge_test_prediction_files(predictions_path: Path, number_of_splits: int = 5) -> Path:
    out_path = predictions_path / 'all_folds_predictions.txt'
    for fold in range(number_of_splits):
        print('Parsing fold %s' % fold)
        predictions_base_path = predictions_path / 'fold-{}'.format(fold)
        predictions_file = predictions_base_path / 'predictions.txt'

        predictions_lines = predictions_file.open(mode='r', encoding='utf8').readlines()

        output_lines = []

        for idx, prediction_line in enumerate(predictions_lines):
            if prediction_line.strip() != '':
                parts = prediction_line.split(' ')
                token, document_id, line_index, gold_label, predicted_label = parts
                output_lines.append(
                    [token, document_id[2:], line_index[2:], gold_label.strip(), predicted_label.strip(), '\n']
                )
            else:
                assert prediction_line.strip() == ''
                output_lines.append(['\n'])

        with out_path.open(mode='a', encoding='utf8') as out_writer:
            for output_line in output_lines:
                out_writer.write(' '.join(output_line))
        out_writer.close()

    return out_path


def convert_transformers_predictions_to_conll(transformers_predictions_path: Union[str, Path],
                                              conll_input_path: Union[str, Path], fold: int, version: str):
    predictions_lines = Path(transformers_predictions_path).open(mode='r', encoding='utf-8').readlines()
    predictions = [line.strip().split() for line in predictions_lines]
    sentences, predictions = load_sentences(Path(conll_input_path), fold=fold, predictions=predictions)
    if fold is not None:
        predictions_path = get_or_create_path(f'predictions/version_{version}/fold-{fold}') / 'predictions.txt'
    else:
        predictions_path = get_or_create_path(f'predictions/version_{version}') / 'predictions.txt'

    assert len(sentences) == len(predictions)
    with predictions_path.open(mode='w', encoding='utf8') as out_file:
        for sentence, predictions in tqdm(zip(sentences, predictions), 'Writing predictions...'):
            for token_label, prediction in zip(sentence, predictions):
                out_file.write(' '.join(
                    [token_label[0], sentence.document_id, f'B-{sentence.line_index}-{sentence.sentence_index}',
                     token_label[-1], prediction]) + '\n')
            out_file.write('\n')


def convert_cross_validation_transformers_predictions_to_conll(cross_validation_path: Union[str, Path], version: str,
                                                               number_of_splits: int = 5):
    datasets_path = Path(f'dataset/version_{version}/entities/')

    for fold in range(number_of_splits):
        transformers_predictions_path = cross_validation_path / f'fold-{fold}' / 'predictions.txt'
        conll_input_path = datasets_path / f'fold-{fold}/test.conll'

        convert_transformers_predictions_to_conll(transformers_predictions_path=transformers_predictions_path,
                                                  conll_input_path=conll_input_path,
                                                  fold=fold, version=version)


def convert_allennlp_predictions_to_conll(allennlp_predictions_path: Union[str, Path],
                                          conll_input_path: Union[str, Path], fold: int, version: str,
                                          dataset_type: str, relations_only: bool = False,
                                          discriminator: str = None) -> Path:
    sentences = load_sentences_and_predictions(conll_test_file_path=Path(conll_input_path),
                                               predictions_file_path=allennlp_predictions_path, fold=fold)
    if fold is not None:
        predictions_path = get_or_create_path(f'predictions/version_{version}/fold-{fold}')
    else:
        predictions_path = get_or_create_path(f'predictions/version_{version}')

    discriminator_suffix = f'_{discriminator}' if discriminator else ''

    predictions_file_name = f'predictions_allennlp_merged_{dataset_type}{discriminator_suffix}_relations.txt' \
        if relations_only else f'predictions_allennlp_merged_{dataset_type}{discriminator_suffix}.txt'
    predictions_path = predictions_path / predictions_file_name

    with predictions_path.open(mode='w', encoding='utf8') as out_file:
        for sentence in tqdm(sentences, 'Writing predictions...'):
            for token_label in sentence:
                if relations_only:
                    if token_label[-1]:
                        predicted_label, gold_label = token_label[4], token_label[2]
                    else:
                        predicted_label, gold_label = 'O', 'O'
                else:
                    predicted_label, gold_label = token_label[4], token_label[2]
                out_file.write(' '.join(
                    [token_label[0], sentence.document_id, f'B-{sentence.line_index}-{sentence.sentence_index}',
                     predicted_label, gold_label]) + '\n')
            out_file.write('\n')

    return predictions_path


def convert_cross_validation_allennlp_predictions_to_conll(version: str, number_of_splits: int = 5,
                                                           relations_only: bool = False, discriminator: str = None):
    predictions_path = get_or_create_path(f'predictions/version_{version}')
    datasets_path = Path(f'dataset/version_{version}/entities/')

    discriminator_suffix = f'_{discriminator}' if discriminator else ''

    for fold in range(number_of_splits):

        for dataset_type in ['dev', 'test']:
            allennlp_predictions_path = predictions_path / f'fold-{fold}' / \
                                        f'predictions_allennlp_{dataset_type}{discriminator_suffix}.txt'
            conll_input_path = datasets_path / f'fold-{fold}/{dataset_type}{discriminator_suffix}.conll'

            convert_allennlp_predictions_to_conll(allennlp_predictions_path=allennlp_predictions_path,
                                                  conll_input_path=conll_input_path,
                                                  fold=fold, version=version, dataset_type=dataset_type,
                                                  relations_only=relations_only, discriminator=discriminator)


def analyze_cross_validation_errors(version: str, predictions_path: str = 'predictions', allennlp_model: bool = False,
                                    relations_only: bool = False, discriminator: str = None,
                                    return_sentences: bool = False):
    discriminator_suffix = f'_{discriminator}' if discriminator else ''

    predictions_path = Path(predictions_path) / f'version_{version}'
    output_report_path = predictions_path / f'errors_cross_validation_allennlp{discriminator_suffix}_{version}.csv' if \
        allennlp_model else f'errors_cross_validation{discriminator_suffix}_{version}.csv'
    all_sentences = load_sentences_per_fold(predictions_path, allennlp_model=allennlp_model,
                                            relations_only=relations_only, discriminator=discriminator)
    sentences_with_error = parse_results(sentences_per_fold=all_sentences)

    pd.DataFrame.from_records(
        [sentence.to_json() for sentence in sentences_with_error]
    ).to_csv(output_report_path, index=False)

    if return_sentences:
        return all_sentences, sentences_with_error


def analyze_errors(version: str):
    predictions_path = get_or_create_path(f'predictions/version_{version}') / 'predictions.txt'
    output_report_path = predictions_path.parent / 'errors_cross_validation.csv'
    fold = 0
    sentences, _ = load_sentences(predictions_path, fold=fold)

    sentences_per_fold = {fold: sentences}
    sentences_with_error = parse_results(sentences_per_fold=sentences_per_fold)

    pd.DataFrame.from_records(
        [sentence.to_json() for sentence in sentences_with_error]
    ).to_csv(output_report_path, index=False)

    return sentences_per_fold, sentences_with_error


if __name__ == '__main__':
    fire.Fire()
