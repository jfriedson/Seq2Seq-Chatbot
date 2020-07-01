# Demo a trained chatbot model

import json
import os
import torch
import torch.optim as optim

from transformer import *

from torch.utils.data import Dataset
import torch.utils.data

from chatbot import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Evaluate function taken from https://github.com/fawazsammani/chatbot-transformer/blob/master/chat.py
def respond(transformer, question, question_mask, max_len, word_map):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    rev_word_map = {v: k for k, v in word_map.items()}
    transformer.eval()
    start_token = word_map['<start>']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)
    
    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim = 1)
        next_word = next_word.item()
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim = 1)   # (1,step+2)
        
    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()
        
    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])
    
    return sentence


# load seq2seq transformer model from save state and fall back to the backup
main_save_file = 'data/nn/chatbot_nn_model.pth.tar'
backup_save_file = 'data/nn/chatbot_nn_model_bak.pth.tar'


if(os.path.exists(main_save_file)):
    transformer.load_state_dict(torch.load(main_save_file))
    print("model loaded from save point")
elif(os.path.exists(backup_save_file)):
    state = torch.load(backup_save_file)
    transformer.load_state_dict(state["model"])
    print("model loaded from training backup point")
else:
    print("train a chatbot first! Run train.py, then come back here")

transformer.eval()
transformer = transformer.to(device)



# main entry
print("bot: Hello.", flush = True)

break_next = False

while(True):
    if not break_next:
        print("you: ", end = '', flush = True)
    user_input = input().lower()

    if break_next:
        break

    if(user_input == "bye"): 
        break_next = True

    # encode user input, and replace words not in vocabulary with the "unknown" identifier
    enc_qus = [vocabulary.get(word, vocabulary['<unk>']) for word in user_input.split()]

    user_input = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    user_input_mask = (user_input!=0).to(device).unsqueeze(1).unsqueeze(1)  

    bot_response = respond(transformer, user_input, user_input_mask, 20, vocabulary);

    print("bot: " + bot_response, flush = True)
