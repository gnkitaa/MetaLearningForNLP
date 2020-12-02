import torch.nn as nn
from transformers import BertModel


class BERTModel(nn.Module):

    def __init__(self, freeze_bert = True):
        super(BERTModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
                
        self.linear_layer = nn.Linear(768, 64)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.Lrelu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.prelu = nn.PReLU()
        

    def forward(self, seq, attn_masks, options):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        contextual_rep_tokens, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        sentence_rep = contextual_rep_tokens[:, 0]
        
        sentence_rep = self.linear_layer(sentence_rep)
        
        if(options.activation):
            if(options.activation=='relu'):
                sentence_rep = self.relu(sentence_rep)
            elif(options.activation=='tanh'):
                sentence_rep = self.tanh(sentence_rep)
            elif(options.activation=='leaky_relu'):
                sentence_rep = self.Lrelu(sentence_rep)
            elif(options.activation=='sigmoid'):
                sentence_rep = self.sigmoid(sentence_rep)
            elif(options.activation=='gelu'):
                sentence_rep = self.gelu(sentence_rep)
            elif(options.activation=='pRelu'):
                sentence_rep = self.prelu(sentence_rep)
            
        return sentence_rep