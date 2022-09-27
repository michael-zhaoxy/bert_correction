import copy
import json
import sys
from io import open
import logging
import torch
from torch import nn
import torch.nn.functional as F
from smbert.common.tokenizers import Tokenizer
from config import *
from transformers import BertModel

def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights

class EcBert(nn.Module):
    def __init__(self,config):
        super(EcBert, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_len = config.max_position_embeddings
        self.num_hidden_layers = config.num_hidden_layers
        self.attention_head_num = config.num_attention_heads
        self.dropout_prob = config.hidden_dropout_prob
        self.attention_head_size = self.hidden_size // self.attention_head_num
        self.tokenizer = Tokenizer(VocabPath)
        self.intermediate_size = config.intermediate_size


        # 申明网络
        self.bert_model = BertModel(config, False)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden_states = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear(hidden_states)
        if labels is not None:
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(output.permute(0, 2, 1), labels)
            return output, hidden_states, loss
        else:
            return output
