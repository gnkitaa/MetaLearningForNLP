import os
from tqdm import tnrange
import torch
import numpy as np


from sample import extract_sample
from loss import *
from utils import save_list_to_file

    
def train(model, optimizer, train_data, options, val_data=None, logger=None):
    '''
    Will consider only seed set samples for +ve class in the support set.
    '''

    if val_data is None:
        best_state = None
    
    train_loss = []
    train_prec = []
    val_loss = []
    val_prec = []
    best_prec = 0
    best_loss = np.inf

    best_model_path = os.path.join(options.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(options.experiment_root, 'last_model.pth')

    train_sent, train_seq, train_mask, train_label = train_data
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                           gamma=options.lr_scheduler_gamma,
                                           step_size=options.lr_scheduler_step)

    if(options.loss=='ProtoTripletLoss'):
        loss_fxn = ProtoTripletLoss
    else:
        raise Exception("Specify a valid loss function")
        
    load_distances(options.distances_file)
    
    for epoch in range(options.max_epochs):
        model.train()
        for episode in tnrange(options.epoch_size, desc="Epoch {:d} train".format(epoch+1)):
            optimizer.zero_grad()
            sample = extract_sample(options.n_way_tr, options.n_support_tr, \
                                    options.n_query_tr, train_seq, train_mask, train_label, train_sent, options.seed_class)
            
            loss, output = loss_fxn(model, sample, options)
            loss.backward()
            optimizer.step()
            train_loss.append(output['loss'])
            train_prec.append(output['prec'])
            
        scheduler.step()
        avg_loss = np.mean(train_loss[-options.epoch_size:])
        avg_prec = np.mean(train_prec[-options.epoch_size:])
        logger.info('Avg Train Loss: {}, Avg Train Precision: {}'.format(avg_loss, avg_prec))
        
        if val_data:
            model.eval()
            count_error = 0
            for episode in tnrange(options.epoch_size, desc="Epoch {:d} validation".format(epoch+1)):
                try:
                    val_sent, val_seq, val_mask, val_label = val_data
                    seed_class_val = 10
                    val_sample = extract_sample(options.n_way_val, options.n_support_val,\
                                                options.n_query_val, val_seq, val_mask, val_label, val_sent)
                    loss, output = loss_fxn(model, val_sample, options)
                    val_loss.append(output['loss'])
                    val_prec.append(output['prec'])
                    
                except Exception as e:
                    logger.error('Error Raised in Validation Sample : '+ str(e))
                    count_error=count_error+1
            
            logger.info('Skipped {} many samples from val data due to unknown processing error'.format(count_error))
            
            avg_loss = np.mean(val_loss[-options.epoch_size:])
            avg_prec = np.mean(val_prec[-options.epoch_size:])
            postfix = ' (Best)' if avg_prec >= best_prec else ' (Best: {})'.format(
                best_prec)
            logger.info('Avg Val Loss: {}, Avg Val Precision: {}{}'.format(
                avg_loss, avg_prec, postfix))
            
            if(options.save_on_loss):
                if avg_loss <= best_loss:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), best_model_path)
                    best_prec = avg_prec
                    best_loss = avg_loss
                    best_state = model.state_dict()
            else:
                if avg_prec >= best_prec:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), best_model_path)
                    best_prec = avg_prec
                    best_state = model.state_dict()

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), last_model_path)


    for name in ['train_loss', 'train_prec', 'val_loss', 'val_prec']:
        save_list_to_file(os.path.join(options.experiment_root,
                                       name + '.txt'), locals()[name])
        