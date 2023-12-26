from analysis_utils import *
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import math
import stanza
import os


colors_err = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
BEST_MODEL_VERSION = '2.3_scratch_1.0'

def get_fold_trees(all_sentences, version, fold, start_time, save=True):
    fold_trees = {}

    # Iterate over subsets
    for subset in ('train', 'dev', 'test'):
        elapsed = round(time.time()-start_time,2)
        print('{} sec elapsed.'.format(elapsed))
        subset_trees = []
        # Iterate over sentences
        for tokens in all_sentences[subset][fold]['tokens']:
            sent = ' '.join(tokens)
            doc = pipeline(sent.lower().replace('-',''))
            const_tree = doc.sentences[0].constituency
            subset_trees.append(const_tree)
        fold_trees[subset] = subset_trees
    if save:
        pkl.dump(fold_trees, open('fold-{}_trees_{}'.format(fold, version), 'wb'))
    return fold_trees


def get_all_trees(all_sentences, version=version, save=True):
    start_time = time.time()
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    for fold in range(5):
        if not os.path.exists('fold-{}_trees_{}'.format(fold, version)):
            print(fold)
            elapsed = round(time.time()-start_time,2)
            print('{} sec elapsed.'.format(elapsed))
            fold_trees = get_fold_trees(all_sentences, version=version, fold=fold, start_time=start_time, save=save)


def plot_diff(diff, filename=None, suffix=''):
    plt.figure(figsize=(10,6))
    font_size = 20
    n_err_types = len(diff.columns.values)-2
    n_ent_types = len(diff.index.tolist())
    x_tick_locs = np.arange(n_ent_types)
    for i, err_type in enumerate(diff.columns.values[:-2]):
        shift = (i-math.floor(n_err_types/2))/(n_err_types+2)
        scores = diff[err_type].values.tolist()
        x_list = [x + shift for x in x_tick_locs]
        plt.bar(x_list, scores, width = 1/(n_err_types+3), label=' '.join([t.capitalize() for t in err_type.split('_')]), color=colors_err[i])
    plt.legend(fontsize=font_size-2)
    plt.ylim(0,0.8)
    plt.xticks(x_tick_locs, labels=[' '.join([t.capitalize() for t in e.split('_')]) for e in diff.index.tolist()], rotation = 45, ha="right", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    yaxis_label = r"$\Delta$(in-train$_{" + suffix + r"}$, out-of-train$_{" + suffix + r"}$)"
    plt.ylabel(yaxis_label, fontsize=font_size, fontstyle='italic')
    plt.tick_params(length=4, width=2)
    plt.tight_layout()
    plt.ylim(-0.5, 1)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    
    
def plot_unique(mean_occs, cnt_type = 'entity', save_path = 'plots/histogram_lexicon.pdf'):
    fontsize=20
    plt.figure(figsize=(10,6))
    labels=[' '.join([t.capitalize() for t in e.split('_')]) for e in mean_occ.keys()]
    plt.bar(labels, mean_occs, color='salmon')

    plt.ylabel(f'Mean count per unique {cnt_type} in train', fontsize=14)
    plt.tick_params(labelsize=fontsize)
    plt.xticks(rotation = 45, ha="right", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_path)
    

def get_mean_occs(entities, cnt_type = 'entity'):
    ent_types = ['intervention', 'outcome', 'population', 'coreference', 'effect_size', 'total']
    n_unique = {f:{k:[] for k in ('in', 'out')} for f in range(5)}
    mean_occ = {ent_type:[] for ent_type in ent_types}
    for f in range(5):
        all_ents_occs = []
        for ent_type in ent_types[:-1]:
            if cnt_type == 'entity':
                test_ents = entities['test'][f][ent_type]
                train_ents = entities['train'][f][ent_type]
                mean_occ[ent_type].append(np.mean([v for v in test_ents.values()]))
                all_ents_occs += entities['test'][f][ent_type].values()
            elif cnt_type == 'POS':
                test_ents = entities_with_pos['test'][f][ent_type]
                train_ents = entities_with_pos['train'][f][ent_type]
                sum_values = [sum(pos_cnts.values()) for pos_cnts in test_ents.values()]
                mean_occ_pos[ent_type].append(np.mean(sum_values))
                all_ents_occs.append(np.mean(sum_values))
        mean_occ['total'].append(np.mean(all_ents_occs))
    for ent_type in ent_types:
        mean_occ[ent_type] = np.mean(mean_occ[ent_type])

    mean_occs = []
    for ent_type in ent_types:
        mean_occs.append(np.mean(mean_occ[ent_type]))
    return mean_occs


if __name__=='__main__':
    # Load sentences
    all_sentences = load_all_sentences(version=BEST_MODEL_VERSION)
    
    # Get all trees from sentences
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    get_all_trees(all_sentences, version=BEST_MODEL_VERSION, save=True)

    # Get entities
    for subset in SUBSETS:
        for fold in range(5):
            all_sentences[subset][fold]['preprocessed_labels'] = preprocess_tags(all_sentences[subset][fold]['labels'])

    entities = get_entities(all_sentences)
    
    # Get entities with POS
    trees = {}
    for fold in range(5):
        trees[fold] = pkl.load(open('fold-{}_trees_{}'.format(fold, version), 'rb'))
    entities_with_pos = get_entities_with_pos(all_sentences, trees)
    
    # Display memotization-based difference of error types based on entities' lexical content
    norm_scores_df_all_mem = get_norm_scores(all_sentences, memorization='mem', entities=entities)
    norm_scores_df_all_not_mem = get_norm_scores(all_sentences, memorization='not mem', entities=entities)
    diff =norm_scores_df_all_mem-norm_scores_df_all_not_mem
    plot_diff(diff, 'plots/err_type_diff_pretrained_total.pdf', 'lex')
    
    # Display memotization-based difference of error types based on entities' POS
    norm_scores_df_all_mem = get_norm_scores(all_sentences, memorization='mem pos', entities=entities_with_pos)
    norm_scores_df_all_not_mem = get_norm_scores(all_sentences, memorization='not mem pos', entities=entities_with_pos)
    diff =norm_scores_df_all_mem-norm_scores_df_all_not_mem
    plot_diff(diff, 'plots/err_type_diff_pretrained_pos_total.pdf', 'POS')
    
    # Get mean nb of occurrences for each entity seen during training
    mean_occs = get_mean_occs(entities, cnt_type = 'entity')
    
    # Plot histogram based on lexical content seen during training
    plot_unique(mean_occs, cnt_type = 'entity', save_path = 'plots/histogram_lexicon.pdf')

    # Get mean nb of occurrences for each POS tag seen during training
    mean_occs_pos = geat_mean_occs(entities_with_pos, cnt_type = 'POS')


    # Plot histogram based on POS sequences seen during training
    plot_unique(mean_occs_pos, cnt_type = 'POS', save_path = 'plots/histogram_lexicon.pdf')