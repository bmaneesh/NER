import random

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn_crfsuite import metrics
from torch.autograd import Variable
# import bi_lstm

from utils import getBatch

bi_lstm = 1
WindowClassifier = 0

flatten = lambda l: [item for sublist in l for item in sublist]

print torch.__version__
print nltk.__version__

cuda = torch.cuda.is_available()
gpus = [0]
if cuda:
    torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor

if __name__ == '__main__':
    try:
        corpus = nltk.corpus.conll2002.iob_sents()
    except LookupError:
        nltk.download()     # invoke nltk installer
    data = []
    # print corpus
    for cor in corpus:
        sent, _, tag = list(zip(*cor))
        data.append([sent, tag])
    # print sent, '\n', tag,'\n', data[-1]

sents, tags = list(zip(*data))
vocab = list(set(flatten(sents)))   # no context???
tagset = list(set(flatten(tags)))
# print vocab[0:8], '\n', tagset

word2index={'<UNK>' : 0, '<DUMMY>' : 1} # dummy token is for start or end of sentence
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}

tag2index = {}
for tag in tagset:
    if tag2index.get(tag) is None:
        tag2index[tag] = len(tag2index)
index2tag={v:k for k, v in tag2index.items()}

WINDOW_SIZE = 2
windows = []

for sample in data:
    dummy = ['<DUMMY>'] * WINDOW_SIZE
    window = list(nltk.ngrams(dummy + list(sample[0]) + dummy, WINDOW_SIZE * 2 + 1))  # generate ngram possibilities for each word
    windows.extend([[list(window[i]), sample[1][i]] for i in range(len(sample[0]))])  # assign the ngrams with labels based on centre word
    # print windows
print len(windows)  # number of samples

random.shuffle(windows)
train_data = windows[:int(len(windows)*0.9)]
test_data = windows[int(len(windows)*0.9):]

def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))

def prepare_tag(tag,tag2index):
    return Variable(LongTensor([tag2index[tag]]))

if WindowClassifier:
    class WindowClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_size, window_size, hidden_size, output_size):
            super(WindowClassifier, self).__init__()
            self.embed = nn.Embedding(vocab_size, embedding_size)
            self.hidden1 = nn.Linear(embedding_size * (window_size * 2 + 1),hidden_size)
            self.hidden2 = nn.Linear(hidden_size, hidden_size)
            self.out_layer = nn.Linear(hidden_size,output_size)
            self.out = nn.ReLU()
            self.softmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(0.3)

        def forward(self,inputs,istraining=False):
            embeds = self.embed(inputs)
            # print embeds.size()
            concated = embeds.view(-1,embeds.size(1)*embeds.size(2))
            hidden1 = self.out(self.hidden1(concated))
            if istraining:
                hidden1 = self.dropout(hidden1)
            hidden2 = self.out(self.hidden2(hidden1))
            if istraining:
                hidden2 = self.dropout(hidden2)
            netout = self.out_layer(hidden2)
            netout = self.softmax(netout)
            return netout

if bi_lstm:
    class bi_lstm(nn.Module):
        def __init__(self, vocab_size, embedding_size, window_size, hidden_size, output_size,batch_size):
            super(bi_lstm, self).__init__()
            self.embed = nn.Embedding(vocab_size, embedding_size)
            self.h_0 = Variable(torch.randn(2, batch_size, hidden_size))
            self.c_0 = Variable(torch.randn(2, batch_size, hidden_size))
            self.hidden1 = nn.LSTM(input_size = embedding_size, num_layers = 2, hidden_size = hidden_size
                                   , bidirectional=True, dropout = 0.3, batch_first = True)
            # self.hidden2 = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, bidirectional = True, dropout = 0.3)
            self.out_layer = nn.Linear(hidden_size*2,output_size)
            # self.out = nn.ReLU()
            self.softmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(0.3)

        def forward(self,inputs,istraining=False):
            embeds = self.embed(inputs)
            # print embeds.size()
            concated = embeds
            hidden1,_ = self.hidden1(concated)
            if istraining:
                hidden1 = hidden1
            # print hidden1[:,-1,:].size()
            netout = self.out_layer(hidden1[:,-1,:])
            netout = self.softmax(netout)
            return netout

BATCH_SIZE = 128
EMBEDDING_SIZE = 50 # x (WINDOW_SIZE*2+1) = 250
HIDDEN_SIZE = 300
EPOCH = 3
LEARNING_RATE = 0.001

# model = WindowClassifier(len(word2index),EMBEDDING_SIZE,WINDOW_SIZE,HIDDEN_SIZE,len(tag2index))
model = bi_lstm(len(word2index),EMBEDDING_SIZE,WINDOW_SIZE,HIDDEN_SIZE,len(tag2index),BATCH_SIZE)
print model
if cuda:
    model = model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

for ep in range(EPOCH):
    losses = []
    acc = []
    for bid,batch in enumerate(getBatch(BATCH_SIZE,train_data)):
        x,y = list(zip(*batch))
        inputs = torch.cat([prepare_sequence(sent, word2index).view(1,-1) for sent in x])
        targets = torch.cat([prepare_tag(tag,tag2index) for tag in y])
        model.zero_grad()
        pred = model(inputs, istraining=True)
        # acc.append(np.where(pred==targets).shape[0])
        # print pred.size()
        loss = loss_function(pred,targets)
        losses.append(loss.data.tolist()[0])
        loss.backward()
        optimizer.step()

        if bid % 1000 == 0:
            print '[%d/%d] mean_loss: %0.2f'%(ep,EPOCH,np.mean(losses))
            losses = []

accuracy = 0
for_f1_score = []
for test in test_data:
    x, y = test[0], test[1]
    input_ = prepare_sequence(x, word2index).view(1, -1)

    i = model(input_).max(1)[1]
    pred = index2tag[i.data.tolist()[0]]
    for_f1_score.append([pred, y])
    if pred == y:
        accuracy += 1

print(accuracy/len(test_data) * 100)

y_pred, y_test = list(zip(*for_f1_score))

sorted_labels = sorted(
    list(set(y_test) - {'O'}),
    key=lambda name: (name[1:], name[0])
)

# sorted_labels

# ['B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

y_pred = [[y] for y in y_pred] # this is because sklearn_crfsuite.metrics function flatten inputs
y_test = [[y] for y in y_test]

print(metrics.flat_classification_report(
    y_test, y_pred, labels = sorted_labels, digits=3
))
