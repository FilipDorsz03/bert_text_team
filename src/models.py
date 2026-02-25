import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from src.constants import MODEL_NAME


class BERTClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.2, pretrain_path=None): #need to give a path to our ssl pretrained bert
        super(BERTClassifier, self).__init__()
        load_from = pretrain_path if pretrain_path else MODEL_NAME
        config = BertConfig.from_pretrained(load_from)
        
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        
        self.bert = BertModel.from_pretrained(load_from, config=config)
        self.output_dim = self.bert.config.hidden_size  # typically 768 for BERT-base
        
        
        #freeze embedding
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        self.fc_final = nn.Linear(self.output_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        

    def forward(self, input_ids, attention_mask):
        # BERT making embedding and analysing
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        #deep_features = outputs.last_hidden_state[:, 0, :]
        deep_features = outputs.pooler_output
        
        dropped_features = self.dropout(deep_features)
        logits = self.fc_final(dropped_features)
        
        return logits, deep_features