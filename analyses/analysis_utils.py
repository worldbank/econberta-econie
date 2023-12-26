from cross_validation_entity_analysis import *
from pathlib import Path
from conll_utils import load_sentences
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as npf
import stanza
import time


LBL = {'O':0,
       'I-intervention':1,
       'I-outcome':2,
       'I-population':3,
       'B-intervention':4,
       'B-outcome':5,
       'B-population':6,
       'B-coreference':7,
       'B-effect_size':8,
       'I-effect_size':9,
       'I-coreference':10,
       'B-None':11,
       'I-None':12
      }


SUBSETS = ['train', 'dev', 'test']


# def get_preds_dev_test(relations_only=True, version = '1.5c-causal_1.0', predictions_path = 'predictions'):
#     """ Loads predictions and labels for both the test and dev sets, along with all 
#     sentences in the test and dev sets as Sentence objects. """ 

#     allennlp_model = True

#     predictions_path = Path(predictions_path) / f'version_{version}'
#     output_report_path = predictions_path / f'errors_cross_validation_allennlp_{version}.csv' if \
#         allennlp_model else f'errors_cross_validation_{version}.csv'
#     all_sentences = load_sentences_per_fold(predictions_path, allennlp_model=allennlp_model,
#                                             relations_only=relations_only, discriminator=None)
#     true_predictions = {}
#     true_labels = {}
#     for subset in ['dev', 'test']:
#         for fold in range(5):
#             true_predictions = [all_sentences[f]['test'][k].predicted_labels for f in range(5) for k in range(len(all_sentences[f]['test']))]
#             true_labels = [all_sentences[f]['test'][k].gold_labels for f in range(5) for k in range(len(all_sentences[f]['test']))]
#     return true_predictions, true_labels, all_sentences


def remove_none(labels):
    new_labels = []
    for l in labels:
        if l.endswith('None'):
            new_labels.append('O')
        else:
            new_labels.append(l)
    return new_labels


def load_train_sentences(version):
    train_sentences = {}
    for fold in range(5):
        fold_train_sentences = {}
        loaded_sentences = load_sentences(f'dataset/version_{version}/entities/fold-{fold}/train.conll')
        fold_train_sentences['tokens'] = [[l[0] for l in s] for s in loaded_sentences]
        fold_train_sentences['document_id'] = [[l[1] for l in s] for s in loaded_sentences]
        fold_train_sentences['line_sentence_id'] = [[l[2] for l in s] for s in loaded_sentences]
        fold_train_sentences['labels'] = [remove_none(iob_bioul([l[3] for l in s])) for s in loaded_sentences]
        train_sentences[fold] = fold_train_sentences
    return train_sentences


def load_all_sentences(version, relations_only=False, predictions_path = 'predictions'):
    """ Loads predictions and labels for the test set, along with all 
    sentences in the test and dev sets as Sentence objects. """ 

    all_sentences = {}
    # Load train sentences
    all_sentences['train'] = load_train_sentences(version)
    # Load dev and test sentences
    allennlp_model = True
    predictions_path = Path(predictions_path) / f'version_{version}'
    output_report_path = predictions_path / f'errors_cross_validation_allennlp_{version}.csv' if \
        allennlp_model else f'errors_cross_validation_{version}.csv'
    dev_test_sentences = load_sentences_per_fold(predictions_path, allennlp_model=allennlp_model,
                                            relations_only=relations_only, discriminator=None)
#     true_predictions = [all_sentences[f]['test'][k].predicted_labels for f in range(5) for k in range(len(all_sentences[f]['test']))]
#     true_labels = [all_sentences[f]['test'][k].gold_labels for f in range(5) for k in range(len(all_sentences[f]['test']))]
    for subset in ('dev', 'test'):
        subset_sentences = {}
        for fold in range(5):
            fold_sentences_obj = dev_test_sentences[fold][subset]
            fold_sentences = {}
            fold_sentences['tokens'] = [s.tokens for s in fold_sentences_obj]
            fold_sentences['document_id'] = [s.document_id for s in fold_sentences_obj]
            fold_sentences['line_sentence_id'] = ['B-'+s.id.split('_')[-1] for s in fold_sentences_obj]
            fold_sentences['labels'] = [remove_none(s.gold_labels) for s in fold_sentences_obj]
            fold_sentences['predictions'] = [s.predicted_labels for s in fold_sentences_obj]
            subset_sentences[fold] = fold_sentences
        all_sentences[subset] = subset_sentences
    return all_sentences


# def get_preds_labels(relations_only=False, version = '1.5c-causal_1.0', predictions_path = 'predictions'):
#     """ Loads predictions and labels for the test set, along with all 
#     sentences in the test and dev sets as Sentence objects. """ 

#     allennlp_model = True

#     predictions_path = Path(predictions_path) / f'version_{version}'
#     output_report_path = predictions_path / f'errors_cross_validation_allennlp_{version}.csv' if \
#         allennlp_model else f'errors_cross_validation_{version}.csv'
#     all_sentences = load_sentences_per_fold(predictions_path, allennlp_model=allennlp_model,
#                                             relations_only=relations_only, discriminator=None)
#     true_predictions = [all_sentences[f]['test'][k].predicted_labels for f in range(5) for k in range(len(all_sentences[f]['test']))]
#     true_labels = [all_sentences[f]['test'][k].gold_labels for f in range(5) for k in range(len(all_sentences[f]['test']))]
#     return true_predictions, true_labels, all_sentences


def get_preds_labels(all_sentences):
    """ Loads predictions and labels for the test set, along with all 
    sentences in the test and dev sets as Sentence objects. """ 
    true_predictions = []
    true_labels = []
    for f in range(5):
        true_predictions += all_sentences['test'][f]['predictions']
        true_labels += all_sentences['test'][f]['labels']
    return true_predictions, true_labels


def get_norm_scores(version, relations_only=False, memorization='all'):
    
    all_sentences = load_all_sentences(relations_only=False, version = version, predictions_path = 'predictions')
    test_true_labels, all_sentences = get_preds_labels(all_sentences)
    if memorization=='all':
        scores_all = match_on_all(test_true_predictions, test_true_labels)
        scores_df_all = convert_res_to_df(scores_all)
    elif memorization=='mem':
        scores_all = match_on_all_memorization(true_predictions, true_labels, entities, all_sentences)[True]
        scores_df_all = convert_res_to_df(scores_all)
    elif memorization=='not mem':
        scores_all = match_on_all_memorization(true_predictions, true_labels, entities, all_sentences)[False]
        scores_df_all = convert_res_to_df(scores_all)

    # print(scores_df_all['total_lab'].sum())
    # scores_df_all

    # Add total line 
    sum_row = scores_df_all.sum(axis=1)
    sum_row.name='Total'
    scores_df_all = scores_df_all.append(sum_row)

    total_scores = {}
    for err_type in scores_all['intervention']:
        total_scores[err_type]=sum([scores_all[ent_type][err_type] for ent_type in scores_all])
    scores_all['total'] = total_scores

    # convert_res_to_df(scores_all)

    norm_scores_all = normalize_results(scores_all)
    norm_scores_df_all = convert_res_to_df(norm_scores_all)
    norm_scores_df_all = norm_scores_df_all.loc[['intervention', 'outcome', 'population', 'coreference', 'effect_size', 'total']]
    return norm_scores_df_all


def preprocess_tags(true_predictions):
    preds = []
    for true_preds in true_predictions:
        cur_preds = []
        cur_ent = 'O'
        for i,p in enumerate(true_preds):
            if p.startswith('U'):
                cur_ent = p[2:]
                end = i
                start = i
                cur_preds.append((cur_ent, start, end))
            elif p.startswith('B'):
                if cur_ent!='O':
                    end = i-1
                cur_ent = p[2:]
                start = i
            elif p.startswith('I'):
                if p[2:]!=cur_ent:
                    if cur_ent != 'O':
                        end = i-1
                        cur_preds.append((cur_ent, start, end))
                    cur_ent = p[2:]
                    start = i
            elif p.startswith('L'):
                if p[2:]!=cur_ent:
                    if cur_ent != 'O':
                        end = i-1
                        cur_preds.append((cur_ent, start, end))
                    cur_ent = p[2:]
                    start = i
                else:
                    end = i
                    cur_preds.append((cur_ent, start, end))
            elif p == 'O':
                if cur_ent != 'O':
                    end = i-1
                cur_ent = 'O'
        preds.append(cur_preds)
    return preds


def preprocess_tags_iob(true_predictions):
    preds = []
    for true_preds in true_predictions:
        cur_preds = []
        cur_ent = 'O'
        for i,p in enumerate(true_preds):
            if p.startswith('B'):
                if cur_ent!='O':
                    end = i-1
                    cur_preds.append((cur_ent, start, end))
                cur_ent = p[2:]
                start = i
            elif p.startswith('I'):
                if p[2:]!=cur_ent:
                    if cur_ent != 'O':
                        end = i-1
                        cur_preds.append((cur_ent, start, end))
                    cur_ent = p[2:]
                    start = i
                end = i
            elif p == 'O':
                if cur_ent != 'O':
                    end = i-1
                    cur_preds.append((cur_ent, start, end))
                cur_ent = 'O'
        preds.append(cur_preds)
    return preds

# def preprocess_train_labs(true_predictions):
#     preds = []
#     for true_preds in true_predictions:
#         cur_preds = []
#         cur_ent = 'O'
#         for i,p in enumerate(true_preds):
#             if p.startswith('B'):
#                 if cur_ent!='O':
#                     end = i-1
#                     cur_preds.append((cur_ent, start, end))
#                 cur_ent = p[2:]
#                 start = i
#             elif p.startswith('I'):
#                 if p[2:]!=cur_ent:
#                     if cur_ent != 'O':
#                         end = i-1
#                         cur_preds.append((cur_ent, start, end))
#                     cur_ent = p[2:]
#                     start = i
#             elif p == 'O':
#                 if cur_ent != 'O':
#                     end = i-1
#                     cur_preds.append((cur_ent, start, end))
#                 cur_ent = 'O'
#         preds.append(cur_preds)
#     return preds


def match_preds_labs(labs, preds, res, seen_label=False, seen_prediction=False):
    if len(preds)==0:
        if len(labs)>0:
            lab_ent, _, _, = labs.pop(0)
            if not seen_label:
                res[lab_ent]['missed_label']+=1
                res[lab_ent]['total_lab']+=1
            return match_preds_labs(labs, preds, res)
        else:
            return res
    elif len(labs)==0:
        pred_ent, _, _ = preds.pop(0)
        if not seen_prediction:
            res[pred_ent]['false_alarm']+=1
            res[pred_ent]['total_pred']+=1
        return match_preds_labs(labs, preds, res)
    else:
        lab_ent, lab_start, lab_end = labs.pop(0)
        pred_ent, pred_start, pred_end = preds.pop(0)
        if (lab_start == pred_start and lab_end == pred_end):
            if lab_ent == pred_ent:
                res[lab_ent]['exact_match']+=1
            else:
                res[lab_ent]['exact_boundary']+=1
            res[lab_ent]['total_lab']+=1
            res[lab_ent]['total_pred']+=1
            return match_preds_labs(labs, preds, res)
        elif pred_end < lab_start:
            if not seen_prediction:
                res[pred_ent]['false_alarm']+=1
                res[pred_ent]['total_pred']+=1
            return match_preds_labs([(lab_ent, lab_start, lab_end)]+labs, preds, res)
        else:
            if (lab_start <= pred_end and pred_end <= lab_end) or (pred_start <= lab_end and lab_end <= pred_end):
                if pred_ent == lab_ent:
                    res[lab_ent]['partial_match']+=1
                else:
                    res[lab_ent]['partial_boundary']+=1
                res[lab_ent]['total_lab']+=1
                res[lab_ent]['total_pred']+=1
                if pred_end==lab_end:
                    return match_preds_labs(labs, preds, res)
                elif pred_end < lab_end:
                    return match_preds_labs([(lab_ent, lab_start, lab_end)]+labs, preds, res, seen_label=True, seen_prediction=False)
                elif lab_end < pred_end:
                    return match_preds_labs(labs, [(pred_ent, pred_start, pred_end)]+preds, res, seen_label=False, seen_prediction=False)
            elif lab_end < pred_start:
                if not seen_label:
                    res[lab_ent]['missed_label']+=1
                    res[lab_ent]['total_lab']+=1
                return match_preds_labs(labs, [(pred_ent, pred_start, pred_end)]+preds, res)

            
def match_preds_labs_length(labs, preds, res, toks, seen_label=False, seen_prediction=False):
    """ Compute matches between a list of predictions and labels in the test set by 
    breaking down based on whether the entity was present in the training set. """
    if len(preds)==0:
        if len(labs)>0:
            lab_ent, lab_start, lab_end, = labs.pop(0)
            ent = ' '.join(toks[lab_start:lab_end+1])
            if not seen_label:
                res[lab_end+1-lab_start][lab_ent]['missed_label']+=1
                res[lab_end+1-lab_start][lab_ent]['total_lab']+=1
            return match_preds_labs_length(labs, preds, res, toks)
        else:
            return res
    elif len(labs)==0:
        pred_ent, pred_start, pred_end = preds.pop(0)
        ent = ' '.join(toks[pred_start:pred_end+1])
        if not seen_prediction:
            res[pred_end+1-pred_start][pred_ent]['false_alarm']+=1
            res[pred_end+1-pred_start][pred_ent]['total_pred']+=1
        return match_preds_labs_length(labs, preds, res, toks)
    else:
        lab_ent, lab_start, lab_end = labs.pop(0)
        pred_ent, pred_start, pred_end = preds.pop(0)
        if (lab_start == pred_start and lab_end == pred_end):
            ent = ' '.join(toks[lab_start:lab_end+1])
            if lab_ent == pred_ent:
                res[lab_end+1-lab_start][lab_ent]['exact_match']+=1
            else:
                res[lab_end+1-lab_start][lab_ent]['exact_boundary']+=1
            res[lab_end+1-lab_start][lab_ent]['total_lab']+=1
            res[lab_end+1-lab_start][lab_ent]['total_pred']+=1
            return match_preds_labs_length(labs, preds, res, toks)
        elif pred_end < lab_start:
            ent = ' '.join(toks[pred_start:pred_end+1])
            if not seen_prediction:
                res[pred_end+1-pred_start][pred_ent]['false_alarm']+=1
                res[pred_end+1-pred_start][pred_ent]['total_pred']+=1
            return match_preds_labs_length([(lab_ent, lab_start, lab_end)]+labs, preds, res, toks)
        else:
            ent = ' '.join(toks[lab_start:lab_end+1])
            if (lab_start <= pred_end and pred_end <= lab_end) or (pred_start <= lab_end and lab_end <= pred_end):
                if pred_ent == lab_ent:
                    res[lab_end+1-lab_start][lab_ent]['partial_match']+=1
                else:
                    res[lab_end+1-lab_start][lab_ent]['partial_boundary']+=1
                res[lab_end+1-lab_start][lab_ent]['total_lab']+=1
                res[lab_end+1-lab_start][lab_ent]['total_pred']+=1
                if pred_end == lab_end:
                    return match_preds_labs_length(labs, preds, res, toks)
                elif lab_end > pred_end:
                    return match_preds_labs_length([(lab_ent, lab_start, lab_end)]+labs, preds, res, toks, seen_label=True, seen_prediction=False)
                elif lab_end < pred_end:
                    return match_preds_labs_length(labs, [(pred_ent, pred_start, pred_end)]+preds, res, toks, seen_label=False, seen_prediction=True)
            elif lab_end < pred_start:
                if not seen_label:
                    res[lab_end+1-lab_start][lab_ent]['missed_label']+=1
                    res[lab_end+1-lab_start][lab_ent]['total_lab']+=1
                return match_preds_labs_length(labs, [(pred_ent, pred_start, pred_end)]+preds, res, toks)            

            
def match_preds_labs_memorization(labs, preds, res, ent_train, toks, seen_label=False, seen_prediction=False):
    """ Compute matches between a list of predictions and labels in the test set by 
    breaking down based on whether the entity was present in the training set. """
    if len(preds)==0:
        if len(labs)>0:
            lab_ent, lab_start, lab_end, = labs.pop(0)
            ent = ' '.join(toks[lab_start:lab_end+1])
            if not seen_label:
                res[ent in ent_train[lab_ent]][lab_ent]['missed_label']+=1
                res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
            return match_preds_labs_memorization(labs, preds, res, ent_train, toks)
        else:
            return res
    elif len(labs)==0:
        pred_ent, pred_start, pred_end = preds.pop(0)
        ent = ' '.join(toks[pred_start:pred_end+1])
        if not seen_prediction:
            res[ent in ent_train[pred_ent]][pred_ent]['false_alarm']+=1
            res[ent in ent_train[pred_ent]][pred_ent]['total_pred']+=1
        return match_preds_labs_memorization(labs, preds, res, ent_train, toks)
    else:
        lab_ent, lab_start, lab_end = labs.pop(0)
        pred_ent, pred_start, pred_end = preds.pop(0)
        if (lab_start == pred_start and lab_end == pred_end):
            ent = ' '.join(toks[lab_start:lab_end+1])
            if lab_ent == pred_ent:
                res[ent in ent_train[lab_ent]][lab_ent]['exact_match']+=1
            else:
                res[ent in ent_train[lab_ent]][lab_ent]['exact_boundary']+=1
            res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
            res[ent in ent_train[lab_ent]][lab_ent]['total_pred']+=1
            return match_preds_labs_memorization(labs, preds, res, ent_train, toks)
        elif pred_end < lab_start:
            ent = ' '.join(toks[pred_start:pred_end+1])
            if not seen_prediction:
                res[ent in ent_train[pred_ent]][pred_ent]['false_alarm']+=1
                res[ent in ent_train[pred_ent]][pred_ent]['total_pred']+=1
            return match_preds_labs_memorization([(lab_ent, lab_start, lab_end)]+labs, preds, res, ent_train, toks)
        else:
            ent = ' '.join(toks[lab_start:lab_end+1])
            if (lab_start <= pred_end and pred_end <= lab_end) or (pred_start <= lab_end and lab_end <= pred_end):
                if pred_ent == lab_ent:
                    res[ent in ent_train[lab_ent]][lab_ent]['partial_match']+=1
                else:
                    res[ent in ent_train[lab_ent]][lab_ent]['partial_boundary']+=1
                res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
                res[ent in ent_train[lab_ent]][lab_ent]['total_pred']+=1
                if pred_end == lab_end:
                    return match_preds_labs_memorization(labs, preds, res, ent_train, toks)
                elif lab_end > pred_end:
                    return match_preds_labs_memorization([(lab_ent, lab_start, lab_end)]+labs, preds, res, ent_train, toks, seen_label=True, seen_prediction=False)
                elif lab_end < pred_end:
                    return match_preds_labs_memorization(labs, [(pred_ent, pred_start, pred_end)]+preds, res, ent_train, toks, seen_label=False, seen_prediction=True)
            elif lab_end < pred_start:
                if not seen_label:
                    res[ent in ent_train[lab_ent]][lab_ent]['missed_label']+=1
                    res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
                return match_preds_labs_memorization(labs, [(pred_ent, pred_start, pred_end)]+preds, res, ent_train, toks)

            
def match_preds_labs_linxi(labs, preds, res, ent_train, toks, seen_label=False, seen_prediction=False):
    if len(preds)==0:
        if len(labs)>0:
            lab_ent, lab_start, lab_end, = labs.pop(0)
            lab_span = ' '.join(toks[lab_start:lab_end+1])
            if not seen_label:
                res[lab_ent]['missed_label'].append((lab_span, 'None', 'None', lab_span in ent_train[lab_ent], lab_start, lab_end, None, None))
#             res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
            return match_preds_labs_linxi(labs, preds, res, ent_train, toks)
        else:
            return res
    elif len(labs)==0:
        pred_ent, pred_start, pred_end = preds.pop(0)
        pred_span = ' '.join(toks[pred_start:pred_end+1])
        if not seen_prediction:
            res['None']['false_alarm'].append(('None', pred_span, pred_ent, pred_span in ent_train[pred_ent], None, None, pred_start, pred_end))
#         res[ent in ent_train[pred_ent]][pred_ent]['total_pred']+=1
        return match_preds_labs_linxi(labs, preds, res, ent_train, toks)
    else:
        lab_ent, lab_start, lab_end = labs.pop(0)
        pred_ent, pred_start, pred_end = preds.pop(0)
        if (lab_start == pred_start and lab_end == pred_end):
            lab_span = ' '.join(toks[lab_start:lab_end+1])
            if lab_ent == pred_ent:
                res[lab_ent]['exact_match'].append((lab_span, lab_span, lab_ent, lab_span in ent_train[lab_ent], lab_start, lab_end, pred_start, pred_end))
            else:
                res[lab_ent]['exact_boundary'].append((lab_span, lab_span, pred_ent, lab_span in ent_train[lab_ent], lab_start, lab_end, pred_start, pred_end))
#             res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
#             res[ent in ent_train[lab_ent]][lab_ent]['total_pred']+=1
            return match_preds_labs_linxi(labs, preds, res, ent_train, toks)
        elif pred_end < lab_start:
            pred_span = ' '.join(toks[pred_start:pred_end+1])
            if not seen_prediction:
                res['None']['false_alarm'].append(('None', pred_span, pred_ent, pred_span in ent_train[pred_ent], None, None, pred_start, pred_end))
#             res[ent in ent_train[pred_ent]][pred_ent]['total_pred']+=1
            return match_preds_labs_linxi([(lab_ent, lab_start, lab_end)]+labs, preds, res, ent_train, toks)
        else:
            lab_span = ' '.join(toks[lab_start:lab_end+1])
            pred_span = ' '.join(toks[pred_start:pred_end+1])
            if (lab_start <= pred_end and pred_end <= lab_end) or (pred_start <= lab_end and lab_end <= pred_end):
#                 ent = ' '.join(toks[lab_start:lab_end+1])
                if pred_ent == lab_ent:
                    res[lab_ent]['partial_match'].append((lab_span, pred_span, lab_ent, lab_span in ent_train[lab_ent], lab_start, lab_end, pred_start, pred_end))
                else:
                    res[lab_ent]['partial_boundary'].append((lab_span, pred_span, pred_ent, lab_span in ent_train[lab_ent], lab_start, lab_end, pred_start, pred_end))
#                 res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
#                 res[ent in ent_train[lab_ent]][lab_ent]['total_pred']+=1
                if pred_end == lab_end:
                    return match_preds_labs_linxi(labs, preds, res, ent_train, toks)
                elif lab_end > pred_end:
                    return match_preds_labs_linxi([(lab_ent, lab_start, lab_end)]+labs, preds, res, ent_train, toks, seen_label=True, seen_prediction=False)
                elif lab_end < pred_end:
                    return match_preds_labs_linxi(labs, [(pred_ent, pred_start, pred_end)]+preds, res, ent_train, toks, seen_label=False, seen_prediction=True)
            elif lab_end < pred_start:
                if not seen_label:
                    res[lab_ent]['missed_label'].append((lab_span, 'None', 'None', lab_span in ent_train[lab_ent], lab_start, lab_end, None, None))
#                 res[ent in ent_train[lab_ent]][lab_ent]['total_lab']+=1
                return match_preds_labs_linxi(labs, [(pred_ent, pred_start, pred_end)]+preds, res, ent_train, toks)

def match_on_all_length(true_predictions, true_labels, all_sentences, max_length):
    """ Compute a counter of matches between all predictions and labels in the test set, in all folds by 
    breaking down based on whether the entity was present in the training set. """
    prep_pred = preprocess_tags(true_predictions)
    prep_lab = preprocess_tags(true_labels)
    res = {l:{k[2:]:{m:0 for m in ['exact_match', 'partial_match', 'exact_boundary', 'partial_boundary', 'missed_label', 'false_alarm', 'total_lab', 'total_pred']} for k in LBL if len(k)>1} for l in range(1,max_length)}
    n_sents = len(prep_pred)//5
    for fold in range(5):
        prep_pred_fold = prep_pred[fold*n_sents:(fold+1)*n_sents]
        prep_lab_fold = prep_lab[fold*n_sents:(fold+1)*n_sents]
        
        for i, labs in enumerate(prep_lab_fold):
            preds = prep_pred_fold[i]
            toks = all_sentences['test'][fold]['tokens'][i]
#             # Sanity check
#             for boundary_tuple in preds:
#                 ent, start, end = boundary_tuple
#                 assert(true_predictions[fold*n_sents+i][start][2:]==ent and true_predictions[fold*n_sents+i][end][2:]==ent)
#             for boundary_tuple in labs:
#                 ent, start, end = boundary_tuple
#                 assert(true_labels[fold*n_sents+i][start][2:]==ent and true_labels[fold*n_sents+i][end][2:]==ent)
            res = match_preds_labs_length(labs, preds, res, toks)
    return res


def match_on_all_memorization(true_predictions, true_labels, entities, all_sentences):
    """ Compute a counter of matches between all predictions and labels in the test set, in all folds by 
    breaking down based on whether the entity was present in the training set. """
    prep_pred = preprocess_tags(true_predictions)
    prep_lab = preprocess_tags(true_labels)
    res = {b:{k[2:]:{m:0 for m in ['exact_match', 'partial_match', 'exact_boundary', 'partial_boundary', 'missed_label', 'false_alarm', 'total_lab', 'total_pred']} for k in LBL if len(k)>1} for b in (True, False)}
    n_sents = len(prep_pred)//5
    for fold in range(5):
        fold_ent_train = entities['train'][fold] # has 5 keys
        prep_pred_fold = prep_pred[fold*n_sents:(fold+1)*n_sents]
        prep_lab_fold = prep_lab[fold*n_sents:(fold+1)*n_sents]
        
        for i, labs in enumerate(prep_lab_fold):
            preds = prep_pred_fold[i]
            toks = all_sentences['test'][fold]['tokens'][i]
#             # Sanity check
#             for boundary_tuple in preds:
#                 ent, start, end = boundary_tuple
#                 assert(true_predictions[fold*n_sents+i][start][2:]==ent and true_predictions[fold*n_sents+i][end][2:]==ent)
#             for boundary_tuple in labs:
#                 ent, start, end = boundary_tuple
#                 assert(true_labels[fold*n_sents+i][start][2:]==ent and true_labels[fold*n_sents+i][end][2:]==ent)
            res = match_preds_labs_memorization(labs, preds, res, fold_ent_train, toks)
    return res

            
def init_match_dict():
    res = {k[2:]:{m:0 for m in ['exact_match', 'partial_match', 'exact_boundary', 'partial_boundary', 'missed_label', 'false_alarm', 'total_lab', 'total_pred']} for k in LBL if len(k)>1}
    return res
    
def match_on_all(true_predictions, true_labels):
    prep_pred = preprocess_tags(true_predictions)
    prep_lab = preprocess_tags(true_labels)
    res = init_match_dict()
    for i, labs in enumerate(prep_lab):
        preds = prep_pred[i]
        # Sanity check
#         for boundary_tuple in preds:
#             ent, start, end = boundary_tuple
#             assert(true_predictions[i][start][2:]==ent and true_predictions[i][end][2:]==ent)
#         for boundary_tuple in labs:
#             ent, start, end = boundary_tuple
#             assert(true_labels[i][start][2:]==ent and true_labels[i][end][2:]==ent)
        res = match_preds_labs(labs, preds, res)
    return res

def convert_res_to_df(res):
    res_to_df={}
    for k in res['intervention']:
        res_to_df[k] = [v[k] for _,v in res.items()]
    res_df = pd.DataFrame(data=res_to_df, columns=[k for k in res['intervention']], index=[k for k in res])
    return res_df

def normalize_results(res):
    norm_res = {}
    for e, e_res in res.items():
        norm_res[e] = {k:v/max(1, e_res['total_lab']) if (k not in ['total_pred', 'false_alarm']) else v/max(1,e_res['total_pred']) for k,v in e_res.items()}
    return norm_res


def get_entities(all_sentences):
    """ Get all entities in each set as a counter.
    TODO : Add dev set """
    entities = {}
    for subset in SUBSETS:
        subset_entities = {}
        for fold in range(5):
            fold_entities = {e_type:Counter() for e_type in ['intervention', 'outcome', 'coreference', 'population', 'effect_size']}
            fold_sentences = all_sentences[subset][fold]
            for i in range(len(fold_sentences['labels'])):
                    toks = fold_sentences['tokens'][i]
                    prep_labels = fold_sentences['preprocessed_labels'][i]
                    for l in prep_labels:
                        ent, start, end = l
                        if ent!='None':
                            fold_entities[ent][' '.join(toks[start:end+1])]+=1
            subset_entities[fold] = fold_entities
        entities[subset] = subset_entities
    return entities


# Plot utils

def get_histo(ent_counter):
    """ Get a histogram from an entity counter. """
    hist = Counter()
    for k,v in ent_counter.items():
        hist[v]+=1
    return hist

def plot_cdf(hist):
    """ Plot cumulative distribution function from a histogram. """
    counts = list(hist.items())
    counts.sort(key=lambda x:x[0], reverse=False)
    vals = [v for v,count in counts]
    counts = [v*count for v,count in counts]
    total = sum(counts)
    heights = [c/total for c in counts]
    cdf_heights = [sum(heights[:i+1]) for i,v in enumerate(heights)]
    plt.bar(vals, cdf_heights)
    plt.xticks(vals)
    plt.show()
    
def get_proportion_seen(entities_train, entities_test):
    """ Get the proportion of entities in test that have been seen in the train set, for each entity type. """
    total = sum(entities_test.values())
    seen = 0
    for e,c in entities_test.items():
        if e in entities_train:
            seen+=entities_test[e]
    return seen/max(1,total)

def plot_proportion_seen(proportion_seen):
    """ Plot the proportion of entities in test that have been seen in the train set, for each entity type. """
    plt.bar(proportion_seen.keys(), proportion_seen.values())
    plt.ylim(0,1)
    plt.show()
    
def plot_all_proportion_seen(all_proportion_seen):
    """ Plot the proportion of entities in test that have been seen in the train set, for each entity type across folds. """
    heights = []
    errs = []
    for e_type in all_proportion_seen[0]:
        e_vals=[]
        for f in range(5):
            e_vals.append(all_proportion_seen[f][e_type])
        heights.append(sum(e_vals)/len(e_vals))
        errs.append(np.std(e_vals)/np.sqrt(len(e_vals)))
    plt.bar(all_proportion_seen[0].keys(), heights, yerr=errs, capsize=5)
    plt.ylim(0,1)
    plt.show()
    
def get_all_proportion_seen(entities, plot=True):
    """ Collects the proportion of entities in test that have been seen in train for each fold.
    Also gets the overlap for entities of length 1 in last fold. 
    Also plots the histogram of number of occurrences for each set and each entity type, for each fold, if plot=True. """
    all_proportion_seen = {}
    overlap_1 = {}
    for fold in range(5):
        proportion_seen = {}
        for e_type in entities['train'][fold]:
            ent_train = entities['train'][fold][e_type]
            ent_test = entities['test'][fold][e_type]
            if plot:
                print("Plotting entities CDF by number of occurrences for type : ", e_type)
                hist_train = get_histo(ent_train)
                plot_cdf(hist_train)
                hist_test = get_histo(ent_test)
                plot_cdf(hist_test)
            overlap_1[e_type] = len([e for e in ent_test if (ent_test[e]==1 and e in ent_train)])/max(1, len([e for e in ent_test if ent_test[e]==1]))
    #         print(sum(ent_train.values()))
    #         print(sum(ent_test.values()))
            proportion_seen[e_type] = get_proportion_seen(ent_train, ent_test)
        if plot:
            print('_'*50)
            plot_proportion_seen(proportion_seen)
            print('_'*50)
        all_proportion_seen[fold] = proportion_seen
    return all_proportion_seen, overlap_1

def get_overlaps(entities):
    """ Get overlaps between entity types in train set for fold 0. """
    n_types = len(entities['train'][0])
    overlaps = np.zeros((n_types,n_types))
    for i, e_type_i in enumerate(entities['train'][0]):
        ents_i = entities['train'][0][e_type_i]
        for j, e_type_j in enumerate(entities['train'][0]):
            ents_j = entities['train'][0][e_type_j]
            overlaps[i,j] = len(set(ents_i.keys()).intersection(set(ents_j.keys())))
    return overlaps

def plot_overlaps(entities):
    """ Plot overlaps between entity types in train set for fold 0 as a matrix. """
    overlaps = get_overlaps(entities)
    fig, ax = plt.subplots()
    im = ax.imshow(overlaps)
    n_types = len(entities['train'][0])
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(n_types))
    ax.set_xticklabels(labels=entities['train'][0].keys())
    ax.set_yticks(np.arange(n_types))
    ax.set_yticklabels(labels=entities['train'][0].keys())
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n_types):
        for j in range(n_types):
            text = ax.text(j, i, overlaps[i, j],
                           ha="center", va="center", color="w")

    plt.show()
    
    
## Syntactic analysis

def get_smallest_constituent(tree, i_min, i_max):
    shift = 0
    if i_max==len(tree.leaf_labels()):
        return tree#, True
    else:
        if not tree.children:
            print('finished')
            return tree#, False
        else:
            for c in tree.children:
                if str(c).startswith('('):
                    if i_min>=shift and i_max<shift+len(c.leaf_labels()):
                        return get_smallest_constituent(c, i_min-shift, i_max-shift)
                    else:
                        shift += len(c.leaf_labels())
            return tree
        
def get_entity_span_counts(all_sentences, true_test_labels, entity_types):
    """ Get the count of each entity span found in the test set """
    entity_spans = {e_type:Counter() for e_type in entity_types}
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    prep_labs = preprocess_tags(true_test_labels)
    n_sents = len(prep_labs)//5
    span_match = {e_type:Counter() for e_type in entity_types}
    for fold in range(5):
        fold_sentences = all_sentences[fold]['test']
        fold_labs = prep_labs[fold*n_sents:(fold+1)*n_sents]
        for i, s in enumerate(fold_sentences):
            if i%100==0:
                print(i)
            sent = ' '.join(s.tokens)
            doc = nlp(sent.lower().replace('-',''))
            const_tree = doc.sentences[0].constituency

            for e_type, i_min, i_max in fold_labs[i]:
                ent_span = ' '.join(s.tokens[i_min:i_max+1])
                lab_const = get_smallest_constituent(const_tree, i_min, i_max)
                const_type = str(lab_const).split(' ')[0][1:]
                entity_spans[e_type][const_type]+=1
    return entity_spans

def plot_spans_bar(cnt):
    """ Util function to plot counters for entity spans. """
    plt.figure(figsize=(7,4))
    cnt_list = list(cnt.items())
    cnt_list.sort(key=lambda x:x[1], reverse=True)
    total = sum(cnt.values())
    tags = [tag for tag, _ in cnt_list]
    heights = [count/total for _, count in cnt_list]
    plt.bar(tags, heights)
    plt.ylim(0,1)
    plt.xlabel('Constituency tag')
    plt.ylabel('Proportion')
    plt.xticks(tags, rotation = 45, ha="center")
    plt.show()
    
    
## Analysis for Luis 

def get_all_num_errors(ent_types, version = '1.6c-causal_1.0'):
    """ Returns a list containing the number of errors for each sentence in all folds of the test set. """
    match_types = ['exact_match', 'exact_boundary', 'partial_match', 'partial_boundary', 'missed_label', 'false_alarm']
    err_types = [match_type for match_type in match_types if match_type!='exact_match']

    all_num_errors = []
    all_sentences = load_all_sentences(relations_only=False, version = version, predictions_path = 'predictions')
    true_predictions, true_labels = get_preds_labels(all_sentences)
    prep_labs = preprocess_tags(true_labels)
    prep_preds = preprocess_tags(true_predictions)
    for i, cur_labs in enumerate(prep_labs):
        cur_preds = prep_preds[i]
        res = init_match_dict()
        res = match_preds_labs(cur_labs, cur_preds, res)
        num_errors = 0
        num_errors += sum([res[ent_type][err_type] for ent_type in ent_types for err_type in err_types])
        all_num_errors.append(num_errors)
    return all_num_errors

def get_cnt(all_num_errors):
    """ Converts list of numbers to counter. """
    cnt = Counter()
    for num_errors in all_num_errors:
        cnt[num_errors]+=1
    return cnt

def plot_error_distribution(error_cnt):
    """ Plots the distribution of number of sentences, for each sentence length. """
    error_cnt_list = list(error_cnt.items())
    error_cnt_list.sort(key=lambda x:x[0])
    num_errors = [e[0] for e in error_cnt_list]
    num_docs = [e[1] for e in error_cnt_list]
    plt.bar(num_errors, num_docs)
    plt.ylabel('# Docs')
    plt.xlabel('# NER Errors')
    plt.show()
    
def plot_error_cdf(error_cnt):
    """ Plots the CDF as above. """ 
    error_cnt_list = list(error_cnt.items())
    error_cnt_list.sort(key=lambda x:x[0])
    num_errors = [e[0] for e in error_cnt_list]
    num_docs = [e[1] for e in error_cnt_list]
    total = sum(num_docs)
    prop_docs = [n/total for n in num_docs]
    cdf = [prop_docs[i]+sum(prop_docs[:i]) for i in range(len(prop_docs))]
    plt.bar(num_errors, cdf)
    plt.ylabel('Cdf Docs')
    plt.xlabel('# NER Errors')
    plt.show()
    
def get_all_num_errors_dict(ent_types, version='1.5c-causal_1.0'):
    """ Returns a dictionary containing the number of errors for each document in (all folds of) the test set. """
    match_types = ['exact_match', 'exact_boundary', 'partial_match', 'partial_boundary', 'missed_label', 'false_alarm']
    err_types = [match_type for match_type in match_types if match_type!='exact_match']

    all_num_errors_dict = {}
    all_sentences = load_all_sentences(relations_only=False, version = version, predictions_path = 'predictions')
    true_predictions, true_labels, all_sentences = get_preds_labels(all_sentences)
    n_per_fold = len(true_labels)//5

    prep_labs = preprocess_tags(true_labels)
    prep_preds = preprocess_tags(true_predictions)
    for i, cur_labs in enumerate(prep_labs):
        cur_preds = prep_preds[i]
        res = init_match_dict()
        res = match_preds_labs(cur_labs, cur_preds, res)
        num_errors = 0
        num_errors += sum([res[ent_type][err_type] for ent_type in ent_types for err_type in err_types])
        fold_id = i//n_per_fold
        id_in_fold = i%n_per_fold

        doc_id = all_sentences[fold_id]['test'][id_in_fold]._document_id
        if doc_id in all_num_errors_dict:
            all_num_errors_dict[doc_id]+=num_errors
        else:
            all_num_errors_dict[doc_id]=num_errors
    return all_num_errors_dict

### Produce dataframe with all predictions

def to_string(tree):
    """ Converts a tree to the string sequence of its leaves. """
    if len(tree.children)==1 and str(tree.children[0])[0]!='(':
        return str(tree.children[0])
    else:
        return ' '.join([to_string(child) for child in tree.children])
    
def create_csv_linxi(true_predictions, true_labels, all_sentences, entities):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    df_dict = {k:[] for k in ('document', #'sector', 
                              'line', 'sentence', 'fold', #'text', 
                              'label_span', 'prediction_span', 'label_entity', 'prediction_entity', 'error_type', 'in_train', 'lab_constituent', 'lab_constituent_type', 'lab_constituent_match', 'pred_constituent', 'pred_constituent_type', 'pred_constituent_match')}


    for subset in ('dev', 'test'):
        for fold in range(5):
            print(fold)
#             for i, s in enumerate(all_sentences[subset][fold]):
            fold_sentences = all_sentences[subset][fold]
            for i in range(len(fold_sentences['labels'])):
                res = {k[2:]:{m:[] for m in ['exact_match', 'partial_match', 'exact_boundary', 'partial_boundary', 'missed_label', 'false_alarm', 'total_lab', 'total_pred']} for k in LBL if len(k)>1}
                prep_labs = preprocess_tags([fold_sentences['labels'][i]])[0]
                prep_preds = preprocess_tags([fold_sentences['predictions'][i]])[0]
                res = match_preds_labs_linxi(prep_labs, prep_preds, res, entities['train'][fold], fold_sentences['tokens'][i])
                
                sent = ' '.join(fold_sentences['tokens'][i])
                doc = nlp(sent.lower().replace('-',''))
                const_tree = doc.sentences[0].constituency
                    
                for lab_ent, lab_res in res.items():
                    for err_type, err_res in lab_res.items():    
                        for i, lab_res in enumerate(err_res):
                            
                            lab_span, pred_span, pred_ent, in_train, lab_start, lab_end, pred_start, pred_end = lab_res
                        
                            if lab_start is not None:
                                lab_const = get_smallest_constituent(const_tree, lab_start, lab_end)
                                lab_const_span = to_string(lab_const)
                                lab_match = (lab_const_span == lab_span)
                                lab_const_type = str(lab_const).split(' ')[0][1:]
                            else:
                                lab_const, lab_const_span, lab_match, lab_const_type = None, None, None, None
                            
                            if pred_start is not None:
                                pred_const = get_smallest_constituent(const_tree, pred_start, pred_end)
                                pred_const_span = to_string(pred_const)
                                pred_match = (pred_const_span == pred_span)
                                pred_const_type = str(pred_const).split(' ')[0][1:]
                            else:
                                pred_const, pred_const_span, pred_match, pred_const_type = None, None, None, None
                            doc_id = fold_sentences['document_id'][i]
                            df_dict['document'].append(doc_id)
#                             df_dict['sector'].append(s._document_id.split('_')[1])
                            line_idx = fold_sentences['line_sentence_id'][i]
                            df_dict['line'].append(line_idx)
                            sent_idx = line_idx.split('_')[0].split('-')[-1]
                            df_dict['sentence'].append(sent_idx)
                            df_dict['fold'].append(fold)
#                             df_dict['text'].append(s._text)
                            df_dict['label_span'].append(lab_span)
                            df_dict['prediction_span'].append(pred_span)
                            df_dict['label_entity'].append(lab_ent)
                            df_dict['prediction_entity'].append(pred_ent)
                            df_dict['error_type'].append(err_type)
                            df_dict['in_train'].append(in_train)
                            lab_const_type = str(lab_const).split(' ')[0][1:]
                            df_dict['lab_constituent'].append(lab_const)
                            df_dict['lab_constituent_type'].append(lab_const_type)
                            df_dict['lab_constituent_match'].append(lab_match)
                            df_dict['pred_constituent'].append(pred_const)
                            df_dict['pred_constituent_type'].append(pred_const_type)
                            df_dict['pred_constituent_match'].append(pred_match)
    df = pd.DataFrame(df_dict)
    return df


def create_csv_from_trees_linxi(true_predictions, true_labels, all_sentences, entities, trees):
#     nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    df_dict = {k:[] for k in ('document', #'sector', 
                              'line', 'sentence', 'fold', #'text', 
                              'label_span', 'prediction_span', 'label_entity', 'prediction_entity', 'error_type', 'in_train', 'lab_constituent', 'lab_constituent_type', 'lab_constituent_match', 'pred_constituent', 'pred_constituent_type', 'pred_constituent_match')}


    for subset in ('dev', 'test'):
        for fold in range(5):
            print(fold)
#             for i, s in enumerate(all_sentences[subset][fold]):
            fold_sentences = all_sentences[subset][fold]
            for i in range(len(fold_sentences['labels'])):
                res = {k[2:]:{m:[] for m in ['exact_match', 'partial_match', 'exact_boundary', 'partial_boundary', 'missed_label', 'false_alarm', 'total_lab', 'total_pred']} for k in LBL if len(k)>1}
                prep_labs = preprocess_tags([fold_sentences['labels'][i]])[0]
                prep_preds = preprocess_tags([fold_sentences['predictions'][i]])[0]
                res = match_preds_labs_linxi(prep_labs, prep_preds, res, entities['train'][fold], fold_sentences['tokens'][i])
                
                sent = ' '.join(fold_sentences['tokens'][i])
#                 doc = nlp(sent.lower().replace('-',''))
#                 const_tree = doc.sentences[0].constituency
                const_tree = trees[fold][subset][i]
                    
                for lab_ent, lab_res in res.items():
                    for err_type, err_res in lab_res.items():    
                        for i, lab_res in enumerate(err_res):
                            
                            lab_span, pred_span, pred_ent, in_train, lab_start, lab_end, pred_start, pred_end = lab_res
                        
                            if lab_start is not None:
                                lab_const = get_smallest_constituent(const_tree, lab_start, lab_end)
                                lab_const_span = to_string(lab_const)
                                lab_match = (lab_const_span == lab_span)
                                lab_const_type = str(lab_const).split(' ')[0][1:]
                            else:
                                lab_const, lab_const_span, lab_match, lab_const_type = None, None, None, None
                            
                            if pred_start is not None:
                                pred_const = get_smallest_constituent(const_tree, pred_start, pred_end)
                                pred_const_span = to_string(pred_const)
                                pred_match = (pred_const_span == pred_span)
                                pred_const_type = str(pred_const).split(' ')[0][1:]
                            else:
                                pred_const, pred_const_span, pred_match, pred_const_type = None, None, None, None
                            doc_id = fold_sentences['document_id'][i]
                            df_dict['document'].append(doc_id)
#                             df_dict['sector'].append(s._document_id.split('_')[1])
                            line_idx = fold_sentences['line_sentence_id'][i]
                            df_dict['line'].append(line_idx)
                            sent_idx = line_idx.split('_')[0].split('-')[-1]
                            df_dict['sentence'].append(sent_idx)
                            df_dict['fold'].append(fold)
#                             df_dict['text'].append(s._text)
                            df_dict['label_span'].append(lab_span)
                            df_dict['prediction_span'].append(pred_span)
                            df_dict['label_entity'].append(lab_ent)
                            df_dict['prediction_entity'].append(pred_ent)
                            df_dict['error_type'].append(err_type)
                            df_dict['in_train'].append(in_train)
                            lab_const_type = str(lab_const).split(' ')[0][1:]
                            df_dict['lab_constituent'].append(lab_const)
                            df_dict['lab_constituent_type'].append(lab_const_type)
                            df_dict['lab_constituent_match'].append(lab_match)
                            df_dict['pred_constituent'].append(pred_const)
                            df_dict['pred_constituent_type'].append(pred_const_type)
                            df_dict['pred_constituent_match'].append(pred_match)
    df = pd.DataFrame(df_dict)
    return df


### Change annotations to fit closest span


def get_smallest_constituent_boundaries(tree, i_min, i_max, p0=0):
    """ Gets smallest constituent in tree which contains [i_min, i_max].
    Returns the subtree, along with its start and end linear left-to-right positions in the original tree. """
    shift = 0
    n_leaves = len(tree.leaf_labels())
    if i_max==n_leaves:
        return tree, (p0, p0+n_leaves-1)
    else:
        if not tree.children:
            print('finished')
            return tree, (p0, p0+n_leaves-1)
        else:
            for c in tree.children:
                if str(c).startswith('('):
                    if i_min>=shift and i_max<shift+len(c.leaf_labels()):
                        return get_smallest_constituent_boundaries(c, i_min-shift, i_max-shift, p0+shift)
                    else:
                        shift += len(c.leaf_labels())
            return tree, (p0, p0+n_leaves-1)
        

def convert_prep_to_gold(prep_labs, sent_len):
    """ Converts a list of preprocessed spans in (e_type, start_pos, end_pos) format to 
    the original annotation format (BIOUL). """
    cur_pos = 0
    gold_labels = []
    for entity_type, lab_start, lab_end in prep_labs:
        while cur_pos < lab_start:
            gold_labels.append('O')
            cur_pos +=1
        if lab_start == lab_end:
            gold_labels.append('U-{}'.format(entity_type))
            cur_pos +=1
        else:
            gold_labels.append('B-{}'.format(entity_type))
            cur_pos+=1
            if lab_end > lab_start + 1:
                for _ in range(lab_start+1, lab_end):
                    gold_labels.append('I-{}'.format(entity_type))
                    cur_pos +=1
            gold_labels.append('L-{}'.format(entity_type))
            cur_pos +=1
    for _ in range(cur_pos, sent_len):
        gold_labels.append('O')
        cur_pos+=1
    return gold_labels


def iob_bioul(tags):
    """
    IOB -> BIOUL
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'U-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'L-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def bioul_iob(tags):
    """
    BIOUL -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            prefix = tag.split('-')[0]
            if prefix in ('B', 'I'):
                new_tags.append(tag)
            elif prefix == 'L':
                new_tags.append(tag.replace('L', 'I'))
            elif prefix == 'U':
                new_tags.append(tag.replace('U', 'B'))
    return new_tags


def merge_labs(prep_labs):
    """ Merges labels that have overlapping boundaries.
    e.g. [..., ('outcome', 15, 15), ('outcome', 14, 20), ...] -> [..., ('outcome', 14, 20), ...]"""
    new_labs = []
    success=True
    for e_type, lab_start, lab_end in prep_labs:
        if not new_labs:
            new_labs.append((e_type, lab_start, lab_end))
        else:
            prev_e_type, prev_lab_start, prev_lab_end = new_labs[-1]
            if prev_lab_end < lab_start:
                new_labs.append((e_type, lab_start, lab_end))
            else:
                if prev_e_type == e_type:
                    new_labs[-1] = (e_type, min(prev_lab_start, lab_start), max(prev_lab_end, lab_end))
                else:
                    print("ERROR : Couldn't merge labels : {}, {}".format((prev_e_type, prev_lab_start, prev_lab_end), (e_type, lab_start, lab_end)))
                    success=False
    return new_labs, success


def print_spans(tokens, labels):
    for e_type, lab_start, lab_end in labels:
        print(e_type, ' : ', ' '.join(tokens[lab_start:lab_end+1]))


def annotate_full_span_for_sent(gold_labels, tokens, const_tree):#, pipeline):
    """ For a single sentence, shifts the boundaries of each label so it fits that of the 
    smallest constituent containing it. 
    Returns the sequence of labels for tokens, in BIOUL format. """
#     start_time = time.time()
    prep_labs = preprocess_tags([gold_labels])[0]
#     print_elapsed(start_time)
#     sent = ' '.join(tokens)
    dash_pos = [p for p, tok in enumerate(tokens) if '-' in tok]
#     doc = pipeline(sent.lower().replace('-',''))
#     print_elapsed(start_time)
#     const_tree = doc.sentences[0].constituency
    const_matched_labs = []
    
    for entity_type, lab_start, lab_end in prep_labs:

        lab_const, lab_const_boundaries = get_smallest_constituent_boundaries(const_tree, lab_start, lab_end)
        lab_const_start, lab_const_end = lab_const_boundaries
        for p in dash_pos:
            if p<lab_const_start:
                lab_const_start+=1
            if p<lab_const_end:
                lab_const_end+=1
                
        const_matched_labs.append((entity_type, lab_const_start, lab_const_end))
#     print_elapsed(start_time)
        
    merged_labs, successful_merge = merge_labs(const_matched_labs)
#     print_elapsed(start_time)
    const_matched_gold_labels = convert_prep_to_gold(merged_labs, len(tokens))
    
#     print_elapsed(start_time)
#     if not successful_merge:
#         print("Old labels:")
#         print_spans(tokens, prep_labs)
#         print("Matched spans:")
#         print_spans(tokens, const_matched_labs)
    return const_matched_gold_labels


def print_elapsed(start_time):
    elapsed = time.time() - start_time
    print(round(elapsed, 2))
    

def annotate_full_span(all_sentences, train_sentences, all_trees):
    """ For a all sentences, shifts the boundaries of each label so it fits that of the 
    smallest constituent containing it. 
    Returns a dictionary containing sentences and their labels for each fold, and each subset, in the following format :
    subset_sentences = {'labels':[], 'tokens':[], 'document_id':[], 'line_sentence_id':[]}.
    Returned labels are in IOB format. """
    
    #pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    new_sentences = {}
    # Iterate over folds
    for fold in range(5):
        fold_new_sentences = {}
        fold_trees = all_trees[fold]
        # Iterate over subsets
        for subset in ('dev', 'test'):
            subset_new_sentences = {'labels':[], 'tokens':[], 'document_id':[], 'line_sentence_id':[]}
            # Iterate over sentences
            subset_trees = fold_trees[subset]
            for i, s in enumerate(all_sentences[fold][subset]):
                start_time = time.time()
                gold_labels, tokens = s.gold_labels, s.tokens
                subset_new_sentences['tokens'].append(s.tokens)
                tree = subset_trees[i]
                const_match_gold_labels = annotate_full_span_for_sent(gold_labels, tokens, tree)#pipeline)
                
#                 print_elapsed(start_time)
            
                subset_new_sentences['document_id'].append([s.document_id for _ in tokens])
                ls_id = 'B-'+s.id.split('_')[-1]
                subset_new_sentences['line_sentence_id'].append([ls_id for _ in tokens])
                subset_new_sentences['labels'].append(bioul_iob(const_match_gold_labels))

#                 print_elapsed(start_time)
#                 print('_')
                
            fold_new_sentences[subset] = subset_new_sentences
            
        train_new_sentences = {'labels':[], 'tokens':[], 'document_id':[], 'line_sentence_id':[]}
        subset_trees = fold_trees['train']
        for i, gold_labels in enumerate(train_sentences[fold]['labels']):
            tokens = train_sentences[fold]['tokens'][i]
            train_new_sentences['tokens'].append(tokens)
            gold_labels_bioul = iob_bioul(gold_labels)
            const_match_gold_labels = annotate_full_span_for_sent(gold_labels_bioul, tokens, tree)
            train_new_sentences['document_id'].append(train_sentences[fold]['document_id'][i])
            train_new_sentences['line_sentence_id'].append(train_sentences[fold]['line_sentence_id'][i])
            train_new_sentences['labels'].append(bioul_iob(const_match_gold_labels))
        fold_new_sentences['train'] = train_new_sentences
        
        new_sentences[fold] = fold_new_sentences
    return new_sentences

def to_conll(data, filename):
    with open(filename, 'w') as txtfile:
        for i, toks in enumerate(data['tokens']):
            labs = data['labels'][i]
            if len(toks)==len(labs):
                doc_ids = data['document_id'][i]
                line_sent_ids = data['line_sentence_id'][i]
                for j, tok in enumerate(toks):
                    lab = labs[j]
                    doc_id = doc_ids[j]
                    line_sent_id = line_sent_ids[j]
                    line = "{} {} {} {}\n".format(tok, doc_id, line_sent_id, lab)
                    txtfile.write(line)
                txtfile.write("\n")
                
                
def get_pos_seq(tree):
    if tree.is_preterminal():
        return [tree.label]
    else:
        preterminals = []
        for t in tree.children:
            preterminals += get_pos_seq(t)
        return preterminals
    
    
def get_entities_with_pos(all_sentences, trees):
    """ Get all entities in each set as a counter.
    TODO : Add dev set """
    entities = {}
    for subset in SUBSETS:
        subset_entities = {}
        for fold in range(5):
            fold_entities = {e_type:{} for e_type in ['intervention', 'outcome', 'coreference', 'population', 'effect_size']}
            fold_sentences = all_sentences[subset][fold]
            fold_trees = trees[fold][subset]
            # Get train entities
            for i in range(len(fold_sentences['labels'])):
                    toks = fold_sentences['tokens'][i]
                    prep_labels = fold_sentences['preprocessed_labels'][i]
                    pos_seq = get_pos_seq(fold_trees[i])
                    for l in prep_labels:
                        ent, start, end = l
                        ent_pos = pos_seq[start:end+1]
                        pos_string = '-'.join(ent_pos)
                        ent_string = ' '.join(toks[start:end+1])
                        if ent!='None':
                            if pos_string not in fold_entities[ent]:
                                fold_entities[ent][pos_string]=Counter()
                            fold_entities[ent][pos_string][ent_string]+=1
            subset_entities[fold] = fold_entities
        entities[subset] = subset_entities
    return entities