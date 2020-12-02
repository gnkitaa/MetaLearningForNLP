#!/usr/bin/env python

import torch
import os
import pandas as pd
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm_notebook
from tqdm import tnrange

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

from utils import *
from data import *
from model import *
from train import train
from test import test

    
def main():
    '''
    Initialize everything and train
    '''
    options = dotdict({})
    theme = 'personalized'
    num_seed = 27
    
    options['dataset_root'] = './Data/'
    options['experiment_root'] = './Models/'
    options['pretrained_model_path'] = './Models/'
    
    options['textfield'] = 'text'
    options['labelfield'] = 'label'
    options['delimiter'] = ','
    
    options['manual_seed'] = 7
    options['device_id'] = 0
    options['cuda'] = True
    
    
    options['maxlen'] = 50
    options['freezeBERT'] = True
    options['activation'] = None
    options['loss'] = 'ProtoTripletLoss'
    options['save_on_loss'] = True
    options['margin'] = 0.8    
    
    options['learning_rate'] = 0.001
    options['max_epochs'] = 5
    options['epoch_size'] = 1000
    options['lr_scheduler_step'] = 2
    options['lr_scheduler_gamma'] = 0.1
    
   
    options['n_way_tr'] = 2
    options['n_support_tr'] = num_seed #has to be num_seed
    options['n_query_tr'] = 100
    options['n_ul_tr'] = 0
    
    
    options['n_way_val'] = 2
    options['n_support_val'] = num_seed
    options['n_query_val'] = 30
    options['n_ul_val'] = 0
    
    
    options['n_way_test'] = 2
    options['n_support_test'] = 5
    options['n_query_test'] = 5
    options['n_ul_test'] = 0
    
    options['test_epoch_size'] = 1
    options['test_small_sample'] = False
    
    print("Are we testing a small sample ?", options.test_small_sample)
    if(options.test_small_sample):
        options['n_way_tr'] = 2        
        options['n_way_val'] = 2
        options['n_way_test'] = 2
        
        options['n_query_tr'] = 5
        options['n_query_val'] = 5
        
        options['max_epochs'] = 2
        options['epoch_size'] = 5
        train_file_name = 'train_small.csv'
        val_file_name = 'val_small.csv'
        test_file_name = 'test_small.tsv'
    else:
        train_file_name = 'train.csv'
        val_file_name = 'val.csv'
        test_file_name = 'test.tsv'
    
    '''-----------------Main Code Begins-----------------'''
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    
    
    '''init logger'''
    logfilename = os.path.join(options.experiment_root, 'logs.log')
    logging.basicConfig(filename=logfilename, level=logging.DEBUG, 
            format='%(asctime)s %(levelname)s %(name)s %(message)s', filemode='w')
    global logger
    logger=logging.getLogger(__name__)

    '''Data'''
    logger.info("Loading Datasets ....")
    train_set_ = ProcessedDataset(filename = os.path.join(options.dataset_root,\
                                                  train_file_name),\
                                                  maxlen = options.maxlen, \
                                                  textfield = options.textfield, \
                                                  labelfield = options.labelfield,\
                                                  delimiter=options.delimiter)

    val_set = ProcessedDataset(filename = os.path.join(options.dataset_root, \
                                                val_file_name), \
                                                maxlen = options.maxlen, \
                                                textfield = options.textfield,\
                                                labelfield = options.labelfield,\
                                                delimiter=options.delimiter)
    
    logger.info("Loaded Datasets")
    
    
    logger.info("Processing Datasets")
    train_data_ = get_data(train_set_)
    val_data = get_data(val_set)
    logger.info("Proceesing completed")
    
    logger.info("Init Model")
    model = BERTModel(freeze_bert = options.freezeBERT)
    model = nn.DataParallel(model) #device_ids=[0,1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = options.learning_rate)
    logger.info("Initialized")

    logger.info('Start Training')
    train(model, optimizer, train_data_, options, val_data, logger)
    logger.info('Training Completed')
    
    logger.info('Testing')
    test(train_data_, val_data, options, logger)
    logger.info('Testing Completed')
    
if __name__ == '__main__':
    main()