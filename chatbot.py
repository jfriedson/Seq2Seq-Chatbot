import json
import torch

from transformer import *


# modify these values for the neural net
attn_heads = 8
enc_dec_layers = 6
model_feats = 512
##########


# model save path
main_save_file = 'data/nn/chatbot_nn_model.pth.tar'
backup_save_file = 'data/nn/chatbot_nn_model_bak.pth.tar'


# program stuff
with open('data/vocabulary.json', 'r') as vocab_file:
    vocabulary = json.load(vocab_file)

transformer = Transformer(model_feats, attn_heads, enc_dec_layers, vocabulary)
