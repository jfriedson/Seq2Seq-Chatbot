# Train the network.  Modified version of https://github.com/fawazsammani/chatbot-transformer/blob/master/train.py

print("you may ignore pytorch deprecation warnings below.")

import json
import os
import torch
import torch.optim as optim

from transformer import *

from torch.utils.data import Dataset
import torch.utils.data

from chatbot import *



# modify these values for training
epochs = 10
learn_rate = 0.0001

backup_after = 10 # mini batches

# if you receive errors about running out of memory:
# close all running processes if possible (they sometimes use gpu memory)
# then, reduce the batch size until the error goes away
# if the number works off the bat, you may want to increase it for faster training and better results!
batch_size = 20

#######################



# both methods below taken from https://github.com/fawazsammani/chatbot-transformer/blob/master/utils.py
class Dataset(Dataset):

    def __init__(self):

        self.pairs = json.load(open('data/pairs_encoded.json'))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
            
        return question, reply

    def __len__(self):
        return self.dataset_size


def create_masks(question, reply_input, reply_target):
    
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    
    question_mask = (question!=0).to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)         # (batch_size, 1, 1, max_words)
     
    reply_input_mask = reply_input!=0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data) 
    reply_input_mask = reply_input_mask.unsqueeze(1) # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target!=0              # (batch_size, max_words)
    
    return question_mask, reply_input_mask, reply_target_mask
######################################################################################


# perplexity function
class LossWithLS(nn.Module):

    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
        
    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))   # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)   # (batch_size * max_words)
        mask = mask.float()
        mask = mask.view(-1)       # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)    # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss
#################################################


# prepare vocabulary and dialogue pairs
train_loader = torch.utils.data.DataLoader(Dataset(),
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)


criterion = LossWithLS(len(vocabulary), 0.1)
##################################################


# prepare pytorch and transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


main_save_file = 'data/nn/chatbot_nn_model.pth.tar'
backup_save_file = 'data/nn/chatbot_nn_model_bak.pth.tar'

trained_epoch = 0
optimizer = torch.optim.Adam(transformer.parameters(), lr=learn_rate, betas=(0.9, 0.98), eps=1e-9)

if(os.path.exists(backup_save_file)):
    state = torch.load(backup_save_file)
    trained_epoch = state["epoch"]
    transformer.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print("model loaded from training backup point")
elif(os.path.exists(main_save_file)):
    transformer.load_state_dict(torch.load(main_save_file))
    print("model loaded from save point")
else:
    print("creating new model to train")

transformer = transformer.to(device)


###########################################



def train(train_loader, transformer, criterion, epoch):
    
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):
        
        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)


        # Prepare Data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # Create mask and add dimensions
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # Get the transformer outputs
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # Loss is directly related to perplexity, but a more simple metric to calculate
        loss = criterion(out, reply_target, reply_target_mask)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        
        sum_loss += loss.item() * samples
        count += samples
        
        # report progress
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}\tPerplexity: {:.3f}".format(epoch+trained_epoch, i, len(train_loader), sum_loss/count, math.exp(sum_loss/count)))

        # save network backup every 1000 dialogues
        if i % (backup_after*100) == 0:
            state = {'epoch': epoch, 'model': transformer.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, backup_save_file)

print("starting training...")
for epoch in range(epochs):
    train(train_loader, transformer, criterion, epoch)
    
    torch.save(transformer.state_dict(), main_save_file)
