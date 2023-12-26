from typing import Union
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from pathlib import Path
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

from confusion_matrix_pretty_print import pretty_plot_confusion_matrix


def plot_confusion_matrix(y_true, y_pred, output_path: Union[Path, str] = None, cmap=plt.cm.Blues, size=20,
                          axis_font_size='x-large',
                          font_size=12, none_class='O'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    classes = get_sorted_labels(all_labels=y_true, none_class=none_class)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    precision = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normal = np.nan_to_num((2 * (precision * recall)) / (precision + recall))
    # cm_normal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm_normal = np.nan_to_num(cm_normal)

    font_prop = fm.FontProperties(size=font_size)
    text_kwargs = dict(fontproperties=font_prop)

    fig, ax = plt.subplots()
    im = ax.imshow(cm_normal, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='')
    ax.set_xlabel('Predicted', **dict(fontsize=axis_font_size))
    ax.set_ylabel('Gold', **dict(fontsize=axis_font_size))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    def format_value(value, percentage):
        return "{:.2%}".format(value) if percentage else format(value, 'd')

    thresh = cm_normal.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if len(classes) > 2 and ((i == (cm.shape[0] - 1)) or (j == (cm.shape[1] - 1))):
                content = format_value(cm[i, j], False)
            else:
                content = format_value(cm[i, j], False) + '\n' + format_value(cm_normal[i, j], True)
            ax.text(j, i, content,
                    ha="center", va="center", fontproperties=font_prop,
                    color="white" if cm_normal[i, j] > thresh else "black")
    fig.tight_layout()

    fig.set_size_inches(size, size)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    # plt.show()


def get_sorted_labels(all_labels, none_class):
    classes = sorted(set(all_labels))
    # print(classes)
    if none_class in classes:
        classes.append(classes.pop(classes.index(none_class)))
    return classes


def add_tag(label_list, tag):
    split = tag.split('-')
    if len(split) > 1:
        label_list.append(split[1])
    else:
        label_list.append(tag)


def print_confusion_matrix(true_labels, predicted_labels, output_name):
    classes = sorted(set(true_labels))
    classes.append(classes.pop(classes.index('O')))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    pretty_plot_confusion_matrix(df_cm, cmap='PuRd', figsize=[12, 12], fmt='.3f', output_name=output_name)


def print_report_from_results(results_file, output_path: Path, cm_size=20, font_size=12):
    predictions_file = Path(results_file).open(mode='r', encoding='utf8')
    lines = predictions_file.readlines()

    true_sentences = []
    predicted_sentences = []
    iob_true_sentences = []
    iob_predicted_sentences = []
    sentence_true_labels = []
    sentence_predicted_labels = []
    sentence_iob_true_labels = []
    sentence_iob_predicted_labels = []
    true_labels = []
    predicted_labels = []
    iob_true_labels = []
    iob_predicted_labels = []
    for line in lines:
        line = line.replace('\n', '')
        split = line.split(' ')
        if len(split) == 1:
            if len(sentence_true_labels) > 0:
                true_sentences.append(sentence_true_labels)
                predicted_sentences.append(sentence_predicted_labels)
                sentence_true_labels = []
                sentence_predicted_labels = []
                iob_true_sentences.append(sentence_iob_true_labels)
                iob_predicted_sentences.append(sentence_iob_predicted_labels)
                sentence_iob_true_labels = []
                sentence_iob_predicted_labels = []
            continue
        elif len(split) == 3:
            token, real_tag, predicted_tag = line.split(' ')
            add_tag(true_labels, real_tag)
            add_tag(sentence_true_labels, real_tag)
            iob_true_labels.append(real_tag)
            sentence_iob_true_labels.append(real_tag)
            add_tag(predicted_labels, predicted_tag)
            add_tag(sentence_predicted_labels, predicted_tag)
            iob_predicted_labels.append(predicted_tag)
            sentence_iob_predicted_labels.append(predicted_tag)

    print('sklearn F1 Score using IOB labels: %s' % f1_score(iob_true_labels, iob_predicted_labels, average='micro'))
    print('sklearn Classification Report using IOB labels:')
    print(classification_report(iob_true_labels, iob_predicted_labels))
    print('sklearn Classification Report using class labels:')
    print(classification_report(true_labels, predicted_labels))

    print('seqeval F1 Score using IOB labels: %s' % seq_f1_score(iob_true_sentences, iob_predicted_sentences))
    print('seqeval Classification Report using IOB labels:')
    print(seq_classification_report(iob_true_sentences, iob_predicted_sentences))
    print('seqeval Classification Report using class labels:')
    print(seq_classification_report(true_sentences, predicted_sentences))

    print('Matplotlib Confusion Matrix:')
    plot_confusion_matrix(true_labels, predicted_labels, output_path, size=cm_size, font_size=font_size)


def save_report_from_results(predictions_path, version, label, confusion_matrix_folder='confusion_matrices',
                             confusion_matrix_size=20):
    confusion_matrix_folder = Path(confusion_matrix_folder)
    if not confusion_matrix_folder.exists():
        confusion_matrix_folder.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = confusion_matrix_folder / 'confusion_matrix_{}_{}.pdf'.format(version, label)
    print_report_from_results(predictions_path, confusion_matrix_path, confusion_matrix_size)


if __name__ == '__main__':
    import fire

    fire.Fire(save_report_from_results)
