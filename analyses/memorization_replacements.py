from evaluate_allennlp_with_conll import *

DATASET_TYPES = ['in_replaced_test_weighted', 'intrain_replaced_test_weighted', 'in_replaced_test_pos_weighted', 'intrain_replaced_test_pos_weighted', 'test']
BEST_MODEL_VERSION = '2.3_scratch_1.0'

def evaluate(model_path: str, version: str, fold: int = None, dataset_type = 'test', cuda_device=-1, batch_size=64, discriminator: str = None):   
    if not model_path.startswith('https'):
        model_path = Path(model_path)

    import_module_and_submodules('allennlp_models')

    if not str(model_path).endswith('.tar.gz'):
        model_path = model_path / 'model.tar.gz'

    archive = load_archive(archive_file=model_path, cuda_device=cuda_device)
    predictor = Predictor.from_archive(archive)

    count = {'count': 0}

    if fold is not None:
        document_folder = get_or_create_path(f'dataset/version_{version}/entities/fold-{fold}')
        predictions_folder = get_or_create_path(f'predictions/version_{version}/fold-{fold}')
        scores_folder = get_or_create_path(f'scores/version_{version}/fold-{fold}')
    else:
        document_folder = get_or_create_path(f'dataset/version_{version}/entities')
        predictions_folder = get_or_create_path(f'predictions/version_{version}')
        scores_folder = get_or_create_path(f'scores/version_{version}')

    discriminator_suffix = f'_{discriminator}' if discriminator else ''

    document_path = document_folder / f'{dataset_type}{discriminator_suffix}.conll'
    predictions_path = predictions_folder / f'predictions_allennlp_{dataset_type}{discriminator_suffix}.txt'
    scores_path = scores_folder / f'scores_allennlp_{dataset_type}{discriminator_suffix}.txt'

    with predictions_path.open(mode='w', encoding='utf8') as out_file:
        for batch in lazy_groups_of(get_instance_data(predictor, document_path), batch_size):
            predict_batch(batch, predictor, count, out_file, raise_oom=False)
    out_file.close()
    print('Finished predicting %d sentences' % count['count'])
    os.system("./%s < %s > %s" % ('conlleval.perl', str(predictions_path), str(scores_path)))
    print(scores_path.open(mode='r', encoding='utf8').read())
    

def evaluate_model(version, dataset_types):
    models_folder = '../models/ner_allennlp_best_{}/'.format(version)
    batch_size = 16 #64
    cuda_device = 0
    discriminator = None

    for dataset_type in dataset_types:
        for fold in range(5):
            model_path = models_folder + 'fold-{}/'.format(fold)
            evaluate(model_path=model_path, version=version, fold=fold, dataset_type=dataset_type, cuda_device=cuda_device, batch_size=batch_size, discriminator=discriminator)
            
import re
def parse_scores(scores_txt):
    re_str = '[0-9]+\.[0-9]+'
    scores = {}
    lines = scores_txt.split('\n')
    overall_scores = re.findall(re_str, lines[1])
    keys = ['accuracy', 'precision', 'recall', 'FB1']
    scores['overall'] = {keys[i]: float(overall_scores[i]) for i in range(len(overall_scores))}
    for line in lines[2:-1]:
        entity = line.split(':')[0]
        entity_scores = re.findall(re_str, line)
        scores[entity] = {keys[i+1]: float(entity_scores[i]) for i in range(len(entity_scores))} 
    return scores

def compute_mean(scores_list):
    mean_scores = {entity_type:{score_type:0 for score_type, score in entity_scores.items()} for entity_type, entity_scores in scores_list[0].items()}
    for scores in scores_list:
        for entity_type, entity_scores in scores.items():
            for score_type, score in entity_scores.items():
                mean_scores[entity_type][score_type]+=score
    for entity_type, entity_scores in scores.items():
        for score_type, score in entity_scores.items():   
            mean_scores[entity_type][score_type]/=len(scores_list)
    return mean_scores
            
def compute_diff(scores, orig_scores):
    scores_diff = {}
    for ent, ent_scores in scores.items():
        scores_diff[ent] = {score_type:round(score-orig_scores[ent][score_type],2) for score_type, score in ent_scores.items()}
    return scores_diff

def print_scores(scores):
    for ent_type, ent_scores in scores.items():
        print(' '.join([f'{ent_type}:']+[f'{score_type}: {score}%;' for score_type, score in ent_scores.items()]))
    print('-'*75)
    
def print_diff(dataset_type, version, orig_type='test'):
    scores_list = []
    orig_scores_list = []
    for fold in range(5):
        scores_folder = get_or_create_path(f'scores/version_{version}/fold-{fold}')
        scores_path = scores_folder / f'scores_allennlp_{dataset_type}.txt'
        scores_txt = scores_path.open(mode='r', encoding='utf8').read()
        scores_list.append(parse_scores(scores_txt))
        orig_path = scores_folder / f'scores_allennlp_{orig_type}.txt'
        orig_scores_txt = orig_path.open(mode='r', encoding='utf8').read()
        orig_scores_list.append(parse_scores(orig_scores_txt))
    mean_scores = compute_mean(scores_list)
    mean_orig_scores = compute_mean(orig_scores_list)
    scores_diff = compute_diff(mean_scores, mean_orig_scores)
    print_scores(scores_diff)
            
if __name__=='__main__':
    evaluate_model(version=BEST_MODEL_VERSION, dataset_types=DATASET_TYPES)
    #evaluate_model(version=BEST_MODEL_VERSION, dataset_types=['in_replaced_test_pos_weighted', 'intrain_replaced_test_pos_weighted', 'test'])
    
    print_diff('in_replaced_test_weighted', version=BEST_MODEL_VERSION)
    print_diff('intrain_replaced_test_weighted', version=BEST_MODEL_VERSION)
    print_diff('in_replaced_test_pos_weighted', version=BEST_MODEL_VERSION)
    print_diff('intrain_replaced_test_pos_weighted', version=BEST_MODEL_VERSION)