import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import math, time
from multiprocessing import Pool
from time import sleep
import numpy as np
from multiprocessing import Pool

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def Intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def TripletLoss(seed, z_query_positive, z_query_negative, options):
    distance_positive = (seed - z_query_positive).pow(2).sum(1)
    distance_negative = (seed - z_query_negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + options.margin)
    loss_val = losses.mean() 
    return loss_val

def Inference(seed, z_query_positive, z_query_negative, n_query):
    z_query = torch.cat((z_query_positive, z_query_negative), dim=0)
    distance_query = (z_query - seed).pow(2).sum(1)
    predicted_sequence = list(torch.argsort(distance_query).detach().cpu().numpy())
    
    actual_pos_points = [i for i in range(n_query)] 
    top_predicted_p = predicted_sequence[0:n_query]
    prec_val_p = 100*len(Intersection(actual_pos_points, top_predicted_p))/min(len(actual_pos_points), n_query)
    return prec_val_p, top_predicted_p

def ProtoTripletLoss(model, sample, options):
    '''Triplet loss where anchor is seed prototype.
    '''
    sample_sequences = sample['sequence'].cuda(options.device_id)
    sample_attention_masks = sample['attention_mask'].cuda(options.device_id)
    
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']

    x_support_seq = sample_sequences[:, :n_support]
    x_query_seq = sample_sequences[:, n_support:]

    x_support_mask = sample_attention_masks[:, :n_support]
    x_query_mask = sample_attention_masks[:, n_support:]

    
    #Concat Seq and Mask of the support and the query set, to get embedding in one call
    x = torch.cat([x_support_seq.contiguous().view(n_way * n_support, *x_support_seq.size()[2:]),
                   x_query_seq.contiguous().view(n_way * n_query, *x_query_seq.size()[2:])], 0).long()

    x_mask = torch.cat([x_support_mask.contiguous().view(n_way * n_support, *x_support_mask.size()[2:]),
                   x_query_mask.contiguous().view(n_way * n_query, *x_query_mask.size()[2:])], 0).long()

    # Get BERT Embedding
    z = model(x, x_mask, options)
    
    z_dim = z.size(-1) #usually 64
    z_support = z[:n_way*n_support].view(n_way, n_support, z_dim)
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    
    z_query = z[n_way*n_support:]
    z_query_reshaped = z_query.view(n_way, n_query, -1)
    z_query_positive = z_query_reshaped[0]
    z_query_negative = z_query_reshaped[1]
    
    num_positives = len(z_query_positive)
    num_negatives = len(z_query_negative)

    
    loss_val = TripletLoss(z_proto[0], z_query_positive, z_query_negative, options)
    prec_val, top_predicted_p = Inference(z_proto[0], z_query_positive, \
                                          z_query_negative, num_positives)
   
    return loss_val, {
        'loss': loss_val.item(),
        'prec': prec_val,
        'y_hat': top_predicted_p
        }