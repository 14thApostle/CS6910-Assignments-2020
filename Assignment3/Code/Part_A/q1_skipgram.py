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
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

WINDOW_SIZE = 2 ## left and right each
EMBEDDING_SIZE = 300
EPOCH = 100
BATCH_SIZE = 2048

class skipgram(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(skipgram, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size

        # Initialize both embedding tables with uniform distribution
        self.embeddings_input.weight.data.uniform_(-1,1)
        self.embeddings_context.weight.data.uniform_(-1,1)

    def forward(self, input_word, context_word):

        # computing out loss
        emb_input = self.embeddings_input(input_word)     # bs, emb_dim
        emb_context = self.embeddings_context(context_word)  # bs, emb_dim
        emb_product = torch.mul(emb_input, emb_context)     # bs, emb_dim       
        emb_product = torch.sum(emb_product, dim=1)          # bs
        out_loss = F.logsigmoid(emb_product)               

        return -(out_loss).mean()
    
    def get_word_vector(self, word_idx):
        word = Variable(torch.LongTensor([word_idx]))
        return self.embeddings(word).view(1, -1)


def get_input_tensor(tensor):
    '''Transform 1D tensor of word indexes to one-hot encoded 2D tensor'''
    size = [*tensor.shape][0]
    inp = torch.zeros(size, vocab_size).scatter_(1, tensor.unsqueeze(1), 1.)
    return Variable(inp).float()

def train_skipgram(data, unique_vocab, word_to_idx, BATCH_SIZE):
    model = skipgram(EMBEDDING_SIZE,len(unique_vocab)+1)
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr = 0.5)
    for epoch in range(EPOCH):
        print(f'Epoch {epoch}')
        total_loss = 0
        for x,y in zip(DataLoader(data[:,0], batch_size=BATCH_SIZE), DataLoader(data[:,1], batch_size=BATCH_SIZE)):
            
            optimizer.zero_grad()
            loss = model(x, y)
            
            loss.backward()
            total_loss+= loss.data
            optimizer.step()    
            
        print(f'Epoch {epoch}, loss {total_loss}')
        if epoch%10==0:
            PATH = "./skipgram_model_final_e{}.pth".format(epoch)
            torch.save(model.state_dict(), PATH)
    PATH = "./skipgram_model_final.pth"
    torch.save(model.state_dict(), PATH)
    return model

# mapping to index
word_to_idx = {w: i+1 for i, w in enumerate(vocab)}
idx_to_word = {ix+1:word for ix, word in enumerate(vocab)}

with open('skipgram_w2i.pickle', 'wb') as handle:
    pickle.dump(word_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('skipgram_i2w.pickle', 'wb') as handle:
    pickle.dump(idx_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)

text = list(cleaned_words)
train_data = []
for i in range(WINDOW_SIZE, len(text) - WINDOW_SIZE):
    data_target = word_to_idx[text[i]]
    for val in text[i-WINDOW_SIZE:i]:
        train_data.append((word_to_idx[val], data_target))
    for val in text[i+1:i+1+WINDOW_SIZE]:
        train_data.append((word_to_idx[val], data_target))

print("Train data: ",len(train_data))

# train model- changed global variable if needed
model = train_skipgram(np.array(train_data), vocab, word_to_idx, BATCH_SIZE)

embeddings = model.embeddings.weight.data.cpu()
np.savez('skipgram_embeddings.npz', name1=embeddings, name2=np.array(list(word_to_idx.keys())))