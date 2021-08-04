from __future__ import print_function
import gensim.downloader as api # package to download text corpus
import nltk # text processing
from nltk.corpus import stopwords
import string
import numpy as np
import pickle
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import csv
import pandas as pd
import matplotlib.pyplot as plt

# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')

# collect all words to be removed
stop = stopwords.words('english') + list(string.punctuation)

actual_words = []
cleaned_words = []
unique_words = set()

# remove stop words
print('removing stop words from text corpus')
for i,words in enumerate(data):
    current_nonstop_words = [w for w in words if w not in stop]
    cleaned_words += current_nonstop_words
    actual_words += words

    for ns in current_nonstop_words:
        unique_words.add(ns)

# print statistics
print(len(actual_words), 'words BEFORE cleaning stop words and punctuations')
print(len(cleaned_words), 'words AFTER cleaning stop words and punctuations')
print('vocabulary size: ', len(unique_words))

# 'cleaned_words' and 'unique_words' to create a word2vec model

vocab = list(unique_words)
print('vocabulary size: ', len(vocab))

WINDOW_SIZE = 4 ## to the left
EMBEDDING_SIZE = 300
EPOCH = 100

class EmbeddingLSTM(nn.Module):
    
    def __init__(self,vocab_size,embedding_size,hidden_dim,n_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_dim,self.n_layers,batch_first=True)

        self.linear1 = nn.Linear(self.hidden_dim, self.vocab_size)
        self.act1 = nn.LogSoftmax(dim = -1)
        
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        embeds = self.embeddings(x)
        lstm_out, hidden = self.lstm(embeds,hidden)
        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)

        out = self.linear1(lstm_out)
        out = self.act1(out)
        out = out.view(batch_size,-1)
        out = out[:, -self.vocab_size:]
        return out, hidden

    def init_hidden(self,batch_size):
        
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return idxs
    # return torch.tensor(idxs, dtype=torch.long)

def train_lstm(train_loader, unique_vocab, word_to_idx):
    lstm = EmbeddingLSTM(len(unique_vocab)+1, EMBEDDING_SIZE, 128, 1)
    if torch.cuda.is_available():
        lstm.cuda()
    h = lstm.init_hidden(BATCH_SIZE)

    loss_fn = nn.NLLLoss()  # loss function
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
    
    for epoch in range(0,EPOCH):
        counter = 0
        total_loss = 0
        for context, target in train_loader: 
            counter += 1 

            h = tuple([each.data for each in h])
            inp_var = context.cuda()
            target_var = target.cuda()
                
            lstm.zero_grad()

            log_prob,h = lstm(inp_var,h)
            loss = loss_fn(log_prob, target_var)
            total_loss += loss.data
            
            loss.backward()
            optimizer.step()

        print("{}/{} loss {:.2f}".format(epoch, EPOCH, total_loss/counter))
        if epoch%10==0:
            PATH = "./lstm_embeds_final_e{}.pth".format(epoch)
            torch.save(model.state_dict(), PATH)

    PATH = "./lstm_embeds_final.pth"
    torch.save(model.state_dict(), PATH)
    return lstm

text = list(cleaned_words)

# mapping to index
word_to_idx = {w: i+1 for i, w in enumerate(vocab)}
idx_to_word = {ix+1:word for ix, word in enumerate(vocab)}

with open('lstm_w2i.pickle', 'wb') as handle:
    pickle.dump(word_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lstm_i2w.pickle', 'wb') as handle:
    pickle.dump(idx_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)


train_x = []
train_y = []
for i in range(WINDOW_SIZE, len(text) - WINDOW_SIZE):
    data_context = text[i-WINDOW_SIZE:i] 
    data_target = text[i]
    train_x.append( make_context_vector(data_context, word_to_idx) )
    train_y.append(word_to_idx[data_target])

train_x = torch.tensor(train_x, dtype=torch.long)
train_y = torch.tensor(train_y, dtype=torch.long)

BATCH_SIZE = 1024
train_data = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

# train model- changed global variable if needed
lstm = train_lstm(train_loader, vocab, word_to_idx)

embeddings = lstm.embeddings.weight.data.cpu()
np.savez('lstm_embeddings.npz', name1=embeddings, name2=np.array(list(word_to_idx.keys())))
