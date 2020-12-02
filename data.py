import torch, os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

class ProcessedDataset(Dataset):

    def __init__(self, filename, maxlen, textfield, labelfield, delimiter='\t'):

        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter = delimiter)
        self.df = self.df.dropna().reset_index(drop=True)
        
        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen
        self.textfield = textfield
        self.labelfield = labelfield

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, self.textfield]
        label = self.df.loc[index, self.labelfield]

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        attn_mask = [1 if token != 0 else 0 for token in tokens_ids]
        return sentence, tokens_ids, attn_mask, label
    

def get_data(data_set):
    _sent = []
    _seq = []
    _mask = []
    _label = []
    
    for i in range(data_set.__len__()):
        sent, seq, mask, label = data_set.__getitem__(i)
        _sent.append(sent)
        _seq.append(seq)
        _mask.append(mask)
        _label.append(label)
    _sent = np.array(_sent)
    _seq = np.array(_seq)
    _mask = np.array(_mask)
    _label = np.array(_label)
    print(_sent.shape, _seq.shape, _mask.shape, _label.shape)
    return (_sent, _seq, _mask, _label)