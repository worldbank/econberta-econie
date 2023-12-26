from analysis_utils import *
import matplotlib.pyplot as plt


BEST_MODEL_VERSION = '2.3_scratch_1.0'
colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
ERR_TYPES = ['exact_match', 'partial_match', 'exact_boundary', 'partial_boundary', 'missed_label', 'false_alarm', 'total_lab', 'total_pred']

def plot_norm_scores_by_length(norm_scores_by_length, filename=None):
    fontsize=24
    plt.figure(figsize=(12,7.5))
    max_length = 15 
    scores = {k:[] for k in norm_scores_by_length[0].keys() if not k.startswith('total')}
    for len_norm_scores in norm_scores_by_length[:max_length]:
        for k, v in len_norm_scores.items():
            if not k.startswith('total'):
                scores[k].append(v)
    cur_bottom = [0 for _ in range(max_length)]
    for i, err_type in enumerate(ERR_TYPES[:-2]):
        heights = scores[err_type][:max_length]
        plt.bar(range(1, max_length+1), heights, bottom=cur_bottom, color=colors[i])
        cur_bottom = [b+heights[i] for i,b in enumerate(cur_bottom)]
    plt.legend(labels=[' '.join([t.capitalize() for t in e.split('_')]) for e in ERR_TYPES[:-2]], fontsize=fontsize-6, loc='upper right', bbox_to_anchor=(1,0.6))
    plt.xlabel('Entity Length', fontsize=fontsize)
    plt.ylabel('Proportion of entities', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.xticks(range(1,max_length+1))
    if filename is not None:
        plt.savefig(filename)
    plt.show()

    
if __name__=='__main__':
    all_sentences = load_all_sentences(version=BEST_MODEL_VERSION)
    test_true_predictions, test_true_labels = get_preds_labels(all_sentences)
    prep_preds = preprocess_tags(test_true_predictions)
    prep_labs = preprocess_tags(test_true_labels)
    max_length = 0

    for preds in prep_preds:
        for pred in preds:
            _, start, end = pred
        max_length = max(max_length, end+2-start)
    for labs in prep_labs:
        for lab in labs:
            _, start, end = lab
            max_length = max(max_length, end+2-start)
    scores_by_length = match_on_all_length(test_true_predictions, test_true_labels, all_sentences, max_length)

    norm_scores_by_length = []
    for l in range(1,max_length):

        len_scores = scores_by_length[l]

        total_scores = {}
        for err_type in len_scores['intervention']:
            total_scores[err_type] = 0
            for ent_type in len_scores:
                total_scores[err_type]+=len_scores[ent_type][err_type]
        len_scores['total'] = total_scores
        norm_scores_all = normalize_results(len_scores)
        norm_scores_df_all = convert_res_to_df(norm_scores_all)
        norm_scores_total = norm_scores_df_all.loc['total']
        norm_scores_by_length.append(norm_scores_total)

    plot_norm_scores_by_length(norm_scores_by_length, 'err_types_by_length.pdf')