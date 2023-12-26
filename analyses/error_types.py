import matplotlib.pyplot as plt
import numpy as np
import math
from analysis_utils import *

VERSION = '2.3'
SLICE = '1.0'
models = ['pretrained', 'scratch', 'mdeberta', 'roberta', 'bert']
versions = [f'{VERSION}_{model}_{SLICE}' for model in models]
model_names = ['EconBERTa-FC', 'EconBERTa-FS', 'mDeBERTa-v3-base', 'RoBERTa-base', 'BERT-base-uncased']

colors = ['dodgerblue', 'springgreen', 'gold', 'goldenrod', 'coral']

def plot_all_scores(all_scores, filename='plots/total_error_types.pdf'):
    plt.figure(figsize=(8,7))
    font_size = 20
    n_types = len(all_scores[versions[0]].columns.values)-2
    x_tick_locs = np.arange(n_types)
    for i, version in enumerate(versions):
        shift = (i-math.floor(len(versions)/2))/(len(versions)+2)
        scores = all_scores[version].loc['total']
        x_list = [x + shift for x in x_tick_locs]
        plt.bar(x_list, scores.values[:-2], width = 1/(len(versions)+3), label=model_names[i], color=colors[i])
    plt.legend(fontsize=font_size)
    plt.ylim(0,0.8)
    plt.ylabel('Proportion of entities', fontsize=font_size)
    plt.xticks(x_tick_locs, labels=[' '.join([t.capitalize() for t in e.split('_')]) for e in scores.keys()[:-2]], rotation = 45, ha="right", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tick_params(length=4, width=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    
if __name__=='__main__':
    per_version_sentences = {}
    for version in versions:
        per_version_sentences[version] = load_all_sentences(version=version)

    fold = None
    all_scores = {}
    for version in versions:
        scores = get_norm_scores(per_version_sentences[version], fold=fold)
        all_scores[version] = scores

    plot_all_scores(all_scores, filename='plots/total_error_types.pdf')