#!/usr/bin/env python
# coding: utf-8


import shutil
from pathlib import Path

import optuna


def load_best_models(models_folder: str = '../models', version: str = '2.3', dataset_slice: str = None):
    """
    :param version: dataset version
    :param models_folder: Parent folder containing the xval folder per dataset slice (ner_allennlp_cv_1.2c_x.x)
    :param dataset_slice: slice of the dataset [0.2, 0.4, 0.6, 0.8, 1.0]
    """
    version_studies = []
    for study in optuna.get_all_study_summaries(storage="sqlite:///allennlp_optuna.db"):
        if str(version) in study.study_name:
            version_studies.append(study)

    version_studies = sorted(version_studies, key=lambda _study: _study.study_name)

    grid_search_folder = 'grid_search_ner_allennlp'
    best_models_folder = 'ner_allennlp_best'

    for version_study in version_studies:
        if dataset_slice is None or version_study.study_name[-3:] == str(dataset_slice):
            study = optuna.load_study(
                storage="sqlite:///allennlp_optuna.db",
                study_name=version_study.study_name
            )
            study_df = study.trials_dataframe().sort_values(by='value', ascending=False)

            for fold in range(5):
                top_idx = 0
                found = False
                while not found:
                    if fold in study_df.params_fold.values:
                        top_fold = study_df[study_df.params_fold == fold].iloc[top_idx].to_dict()
                        models_folder = Path(models_folder)
                        source_path = models_folder / f'{grid_search_folder}_{version}' / f'trial_{top_fold["number"]}'
                        if not source_path.exists():
                            # Ran multiple gridsearch simultaneously in the same database, so there's a folder for each
                            # one
                            top_idx += 1
                            continue
                        else:
                            found = True
                        target_path = models_folder / f'{best_models_folder}_{version}' / f'fold-{fold}/'

                        if target_path.exists():
                            shutil.rmtree(f'{str(target_path)}/')
                        target_path.mkdir(parents=True, exist_ok=True)

                        for file in source_path.iterdir():
                            if file.is_dir():
                                shutil.copytree(file, target_path / file.name)
                            else:
                                if file.suffix != '.th':
                                    shutil.copy(file, target_path)
                    else:
                        print(f"WARNING : Fold {fold} not found in study")

    # Prints the command for evaluating the best models in terms of scores, saving the predictions for each of them for
    # the dev and test datasets, for each fold
    print(f'./evaluate_ner_allennlp_cv.sh {version}_{dataset_slice} {str(models_folder)} {best_models_folder}')


if __name__ == '__main__':
    import fire

    fire.Fire(load_best_models)
