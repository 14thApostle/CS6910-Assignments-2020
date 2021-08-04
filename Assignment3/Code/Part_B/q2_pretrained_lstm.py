from __future__ import print_function
import gensim.downloader as api # package to download text corpus
import nltk # text processing
from nltk.corpus import stopwords
import string
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchtext

import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')

# collect all words to be removed
punctuations = list(string.punctuation)
stop = stopwords.words('english') + list(string.punctuation)

def make_vector(sentence,word_to_idx):
    sentence_vector = []
    for word in sentence.split():
        if word in word_to_idx.keys():
            sentence_vector.append(word_to_idx[word])
        else:
            sentence_vector.append(0)
    return sentence_vector

def remove_punctuations(sentence, punctuations):
    alts = ['\/','-']
    for alt in alts:
        if alt in sentence:
            sentence = sentence.replace(alt,' ')
    final = "".join(u for u in sentence if u not in punctuations)
    return final

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review        
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return torch.tensor(features, dtype=torch.long)

def make_vocab(filename,train):
    tsv_file = open(filename)

    read_tsv = csv.reader(tsv_file, delimiter="\t")
    headers = next(read_tsv, None)

    unique_words = set()
    sentences = []
    sentiments = []
    for i,row in enumerate(read_tsv):
        sentence, sentiment = row[2],row[3]
        # remove punctuation
        sentence = remove_punctuations(sentence, punctuations).lower().split()
        # remove stop words
        sentence = " ".join([w for w in sentence if w not in stop])

        words = sentence.split()
        # remove very small and very large reviews
        if (len(words)>0 and len(words)<15) or not train:
            for word in words:
                unique_words.add(word)
            sentences.append(sentence)
            sentiments.append(int(sentiment))
    return list(unique_words),(sentences,sentiments)

def make_dataset(sentences, sentiments, word_to_idx, SEQ_LENGTH = 8, BATCH_SIZE = 50):

    train_x = [make_vector(sentence, word_to_idx) for sentence in sentences]
    ## pad to const seq_length
    train_x = pad_features(train_x,SEQ_LENGTH).cuda()
    train_y = torch.tensor(sentiments, dtype=torch.long).cuda()

    # create datasets
    train_data = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    return train_loader

BATCH_SIZE = 2048
SEQ_LENGTH = 8
EMBEDDING_SIZE = 300

## Train
vocab,(sentences,sentiments) = make_vocab("Data/train.csv",train=True)
word_to_idx = {w: i+1 for i, w in enumerate(vocab)}
idx_to_word = {ix+1:word for ix, word in enumerate(vocab)}
print("Vocab size", len(vocab))

train_loader = make_dataset(sentences, sentiments, word_to_idx, SEQ_LENGTH, BATCH_SIZE)

## Validation
_,(val_sentences,val_sentiments) = make_vocab("Data/val.csv",train=False)
val_loader = make_dataset(val_sentences, val_sentiments, word_to_idx, SEQ_LENGTH, BATCH_SIZE)

# from torchtext.vocab import GloVe
# embedding = GloVe()
from torchtext.vocab import FastText
embedding = FastText('en')
# from torchtext.vocab import FastText
# embedding = FastText('simple')

matrix_len = len(vocab) + 1
weights_matrix = np.zeros((matrix_len, EMBEDDING_SIZE))
words_found = 0

for word,i in word_to_idx.items():
    if embedding[word].sum()==0:
        # No embedding available
        weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_SIZE, ))
    else:
        words_found += 1
        weights_matrix[i] = embedding[word]

## For 'OOV' words
weights_matrix[0] = np.random.normal(scale=0.6, size=(EMBEDDING_SIZE, ))
weights_matrix = torch.from_numpy(weights_matrix).float()
print("Found {} out of {} words".format(words_found,len(vocab)))

class SentimentLSTM(nn.Module):
    
    def __init__(self,corpus_size,output_size,embedd_dim,hidden_dim,n_layers):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding.from_pretrained(weights_matrix)
        self.lstm = nn.LSTM(embedd_dim, hidden_dim,n_layers,batch_first=True)

        self.fc = nn.Linear(hidden_dim,output_size)
        self.act = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds,hidden)
        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)

        out = self.fc(lstm_out)
        out = self.act(out)
        out = out.view(batch_size,-1)
        out = out[:,-self.output_size:]
        return out, hidden

    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# Instantiate the model w/ hyperparams
vocab_size = len(vocab)+1 # +1 for the 0 padding
output_size = 5
embedding_dim = EMBEDDING_SIZE
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

net.train()
clip=5
epochs = 200
lr=0.01

def criterion(input, target):
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    return l.mean()

## Whether we need to freeze the embedding or not
freeze_embeddings = True
if freeze_embeddings:
    net.embedding.weight.requires_grad = False
optimizer = torch.optim.Adam([ param for param in net.parameters() if param.requires_grad == True], lr=lr)

net.cuda()
losses = []
train_accs=[]
val_accs = []

os.makedirs("./LSTM1/",exist_ok = True)

for e in range(epochs):
    PATH = './LSTM1/model_{}'.format(e)
    # initialize hidden state
    h = net.init_hidden(BATCH_SIZE)
    running_loss = 0.0
    running_acc = 0.0
    # batch loop
    for idx,(inputs, labels) in enumerate(train_loader):
        
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero accumulated gradients
        optimizer.zero_grad()

        # get the output from the model
        output, h = net(inputs.cuda(), h)
        labels=torch.nn.functional.one_hot(labels, num_classes=5).cuda()
        # calculate the loss and perform backprop
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.cpu().detach().numpy()
        running_acc += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
 
    print("Epoch: {}/{}...".format(e+1, epochs),
          "Loss: {:.6f}...".format((running_loss/(idx+1))),
          "Train acc: {}".format((running_acc/(idx+1))) )
    
    losses.append(float(running_loss/(idx+1)))
    train_accs.append(float(running_acc/(idx+1)))

    ## do validation
    h = net.init_hidden(BATCH_SIZE)        
    running_acc = 0.0

    for idx,(inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        # get the output from the model
        output, h = net(inputs.cuda(), h)
        labels = torch.nn.functional.one_hot(labels, num_classes=5).cuda()
        running_acc += (output.argmax(dim=1)==labels.argmax(dim=1)).float().mean()
    print(f'Val acc:{running_acc/(idx+1)}')
    val_accs.append(running_acc/(idx+1))
    
    torch.save(net.state_dict(), PATH)

train_acc_all = train_accs
val_acc_all = val_accs
plt.figure()
plt.plot(np.arange(len(train_acc_all)),train_acc_all)
plt.plot(np.arange(len(train_acc_all)),val_acc_all)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.yticks(np.arange(0,1.1,0.1))
plt.savefig('{}.png'.format("train_fast_en"))
plt.show()


## Testing
net.eval()
net.cuda()
_,(test_sentences,test_sentiments) = make_vocab("Data/test.csv", train=False)

test_x = [make_vector(sentence, word_to_idx) for sentence in test_sentences]
test_x = pad_features(test_x,SEQ_LENGTH)
test_y = torch.tensor(test_sentiments, dtype=torch.long).cuda()

confusion_matrix = np.zeros((5,5))
h = net.init_hidden(1)
acc = 0
for x,y in zip(test_x,test_y):
    h = tuple([each.data for each in h])
    x,y = x.unsqueeze(0).cuda(),y.cuda()
                    
    # get the output from the model
    output, h = net(x, h)
    labels = torch.nn.functional.one_hot(y, num_classes=5).cuda()
    acc += (output.argmax()==labels.argmax()).float().mean()
    
    confusion_matrix[output.argmax()][labels.argmax()] += 1
    
print("Total test accuracy: ", acc/len(test_y))
class_wise_total = np.sum(confusion_matrix,axis=0)

for i in range(5):
    print("Sentiment - ", i," - accuracy - ",(confusion_matrix[i][i]/class_wise_total[i])*100)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix, annot=True)

plt.title("Confusion Matrix")
plt.xlabel('True label') 
plt.ylabel('Predicted') 

plt.savefig('{}.png'.format("cm_fast_en"))
plt.show()