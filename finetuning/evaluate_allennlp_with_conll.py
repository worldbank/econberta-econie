# coding: utf-8

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import lazy_groups_of, import_module_and_submodules
from typing import List, Iterator, Dict
from allennlp.data import Instance
from pathlib import Path
from utils import get_or_create_path
from cross_validation_entity_analysis import convert_allennlp_predictions_to_conll
from confusion_matrix_utils import save_report_from_results

import os
import fire
import torch


def get_instance_data(predictor, document_path) -> Iterator[Instance]:
    yield from predictor._dataset_reader.read(document_path)


def predict_instances(predictor, batch_data: List[Instance]) -> Iterator[str]:
    yield predictor.predict_batch_instance(batch_data)


def predict_batch(batch: List[Instance], predictor: Predictor, count: Dict[str, int], out_file, raise_oom: bool,
                  category_only: str = None):
    try:
        for _, results in zip(batch, predict_instances(predictor, batch)):
            for idx, result in enumerate(results):
                count['count'] += 1
                real_sentence = batch[idx]
                real_tags = real_sentence.fields['tags'].labels
                words = result['words']
                predicted_labels = result['tags']
                for word_idx, (word, tag) in enumerate(zip(words, predicted_labels)):
                    if category_only is not None:
                        if tag != 'O' and tag.endswith(category_only):
                            write_tag = tag
                        else:
                            # When filtering for a particular category, ignore the predictions of others.
                            write_tag = 'O'
                    else:
                        write_tag = tag

                    out_file.write(' '.join([word, real_tags[word_idx], write_tag]) + '\n')

                out_file.write('\n')
                if count['count'] % 200 == 0:
                    print('Predicted %d sentences' % count['count'])

    except RuntimeError as e:
        if 'out of memory' in str(e) and not raise_oom:
            new_batch_size = int(len(batch) / 2)
            print('| WARNING: ran out of memory, retrying with batch size %d' % new_batch_size)
            for p in predictor._model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            for sub_batch in lazy_groups_of(iter(batch), new_batch_size):
                if new_batch_size == 1:
                    # Last attempt with the minimum batch size
                    predict_batch(batch=sub_batch, predictor=predictor, count=count, out_file=out_file, raise_oom=True,
                                  category_only=category_only)
                else:
                    # Try again with a lower batch size
                    predict_batch(batch=sub_batch, predictor=predictor, count=count, out_file=out_file, raise_oom=False,
                                  category_only=category_only)
        else:
            raise e


def evaluate(model_path: str, version: str, dataset_folder: str, dataset_name: str = None, fold: int = None,
             cuda_device: int = -1, batch_size: int = 16, seed: int = None):
    if not model_path.startswith('https'):
        model_path = Path(model_path)

    import_module_and_submodules('allennlp_models')

    if not str(model_path).endswith('.tar.gz'):
        model_path = model_path / 'model.tar.gz'

    archive = load_archive(archive_file=model_path, cuda_device=cuda_device)
    predictor = Predictor.from_archive(archive)

    count = {'count': 0}

    if fold is not None:
        document_folder = Path(dataset_folder) / f'fold-{fold}'
        predictions_folder = get_or_create_path(f'predictions/version_{version}/fold-{fold}')
        scores_folder = get_or_create_path(f'scores/version_{version}/fold-{fold}')
        confusion_matrix_folder = get_or_create_path(f'confusion_matrices/version_{version}/fold-{fold}')
    else:
        document_folder = Path(dataset_folder)
        predictions_folder = get_or_create_path(f'predictions/version_{version}')
        scores_folder = get_or_create_path(f'scores/version_{version}')
        confusion_matrix_folder = get_or_create_path(f'confusion_matrices/version_{version}')

    if seed:
        predictions_folder = get_or_create_path(predictions_folder / f'seed-{seed}')
        scores_folder = get_or_create_path(scores_folder / f'seed-{seed}')
        confusion_matrix_folder = confusion_matrix_folder / f'seed-{seed}'

    if dataset_name is not None:
        evaluate_dataset(batch_size=batch_size, confusion_matrix_folder=confusion_matrix_folder, count=count,
                         dataset_type=dataset_name, document_folder=document_folder, fold=fold,
                         predictions_folder=predictions_folder, scores_folder=scores_folder, predictor=predictor,
                         version=version)

    else:
        for dataset_type in ['dev', 'test']:
            evaluate_dataset(batch_size=batch_size, confusion_matrix_folder=confusion_matrix_folder, count=count,
                             dataset_type=dataset_type, document_folder=document_folder, fold=fold,
                             predictions_folder=predictions_folder, scores_folder=scores_folder, predictor=predictor,
                             version=version)


def evaluate_dataset(batch_size: int, confusion_matrix_folder: Path, count: Dict[str, int], dataset_type: str,
                     document_folder: Path, predictions_folder: Path, scores_folder: Path, predictor: Predictor,
                     version: str = None, fold: int = None, category_only: str = None):
    document_path = document_folder / f'{dataset_type}.conll'
    predictions_path = predictions_folder / f'predictions_allennlp_{dataset_type}.txt'
    scores_path = scores_folder / f'scores_allennlp_{dataset_type}.txt'
    with predictions_path.open(mode='w', encoding='utf8') as out_file:
        for batch in lazy_groups_of(get_instance_data(predictor, document_path), batch_size):
            predict_batch(batch=batch, predictor=predictor, count=count, out_file=out_file, raise_oom=False,
                          category_only=category_only)
    out_file.close()
    print('Finished predicting %d sentences' % count['count'])
    os.system("./%s < %s > %s" % ('conlleval.perl', str(predictions_path), str(scores_path)))
    print(scores_path.open(mode='r', encoding='utf8').read())
    converted_predictions_path = convert_allennlp_predictions_to_conll(
        allennlp_predictions_path=predictions_path,
        conll_input_path=document_path,
        fold=fold, version=version,
        dataset_type=dataset_type)
    converted_scores_path = scores_folder / converted_predictions_path.name.replace('predictions', 'scores')
    print('Calculating score for merged dataset using full dataset')
    os.system("./%s < %s > %s" % ('conlleval.perl', str(converted_predictions_path),
                                  str(converted_scores_path)))
    print(converted_scores_path.open(mode='r', encoding='utf8').read())
    save_report_from_results(predictions_path=predictions_path, confusion_matrix_folder=confusion_matrix_folder,
                             version=version, label=f"{dataset_type}")


if __name__ == '__main__':
    fire.Fire(evaluate)
