from tqdm import tnrange
import numpy as np
import torch, os
import torch.nn as nn
from loss import *
from model import *

def load_ckp(checkpoint_fpath, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint)
    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

def getstats(data, logger=None):
    query_sentence, query_seq, query_mask, query_label = data
    logger.info("Positives : {}".format(len(np.where(query_label==1)[0])))
    logger.info("Negatives : {}".format(len(np.where(query_label==-1)[0])))
    
    query_label_seed = query_label[query_is_seed==1]
    query_label_non_seed = query_label[query_is_seed!=1]
          
    logger.info("Seed Positives : {}".format(len(np.where(query_label_seed==1)[0])))
    logger.info("Non Seed Positives : {}".format(len(np.where(query_label_non_seed==1)[0])))

def test(train_data, test_data, options, logger=None):   
    
    '''Load the best save model'''
    best_model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model = BERTModel(freeze_bert = options.freezeBERT)
    model = load_ckp(best_model_path, model)
    
    '''Get Seed Embeddings from train Data'''
    _sentence, _seq, _mask, _label = train_data

    pos_sentence, pos_seq, pos_mask = _sentence[_label==1], \
                                                _seq[_label==1], \
                                                _mask[_label==1]

    seed_sentence, seed_seq, seed_mask = pos_sentence[pos_is_seed==1], pos_seq[pos_is_seed==1], \
                                                pos_mask[pos_is_seed==1]
    

    '''Find seed prototype'''
    seed_seq = torch.from_numpy(seed_seq).float().cuda(options.device_id).long()
    seed_mask = torch.from_numpy(seed_mask).int().cuda(options.device_id).long()
    z_seed = model(seed_seq, seed_mask, options)
    z_seed_proto = z_seed.mean(0)
    
    '''query embeddings'''
    query_sentence, query_seq, query_mask, query_label = test_data
    getstats(test_data, logger)
    
    query_seq = torch.from_numpy(query_seq).float().cuda(options.device_id).long()
    query_mask = torch.from_numpy(query_mask).int().cuda(options.device_id).long()

    z_tests = []
    step = 50
    for i in range(0, len(query_seq), step):
        z_test = model(query_seq[i:i+step], query_mask[i:i+step], options)
        z_tests.append(z_test)

    z_test_all = torch.stack(z_tests[:-1])
    z_test_all = z_test_all.view(z_test_all.shape[0]*z_test_all.shape[1], z_test_all.shape[2])
    z_test_all = torch.cat((z_test_all, z_tests[-1]), 0)
    
    z_query_positive = z_test_all[query_label==1]
    z_query_negative = z_test_all[query_label==-1]
    
    num_positives = len(z_query_positive)
    num_negatives = len(z_query_negative)
    
    del z_tests, z_test
    torch.cuda.empty_cache()
    
    '''Inference'''
    prec_val, top_predicted_p = Inference(z_seed_proto, z_query_positive, \
                                          z_query_negative, num_positives)
        
   
    logger.info('Test Prec@{} : {}'.format(num_positives, prec_val))