# %% [code]
import wandb
import seaborn as sns
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import gc
import random
import numpy as np 
import pandas as pd
parser = argparse.ArgumentParser(description='Model for multi-class classification')
parser.add_argument('-wp','--wandb_project', default="CS6910_DLassignment_3", required=False,metavar="", type=str, help=' ')
parser.add_argument('-we','--wandb_entity', default="cs22m031", required=False,metavar="", type=str, help='')
parser.add_argument('-e','--epochs', default=10, required=False,metavar="", type=int, help=' ')
parser.add_argument('-b','--batchsize', default=1024, required=False,metavar="", type=int, help=' ')
parser.add_argument('-hidden','--hidden_size', default=128, required=False,metavar="", type=int, help=' ')
parser.add_argument('-embed','--embedding_size',default = 128,required=False,metavar="", type=int, help=' ')
parser.add_argument('-cell','--cell_type',default = 'LSTM',required=False,metavar="", type=str, help=' ',choices = ['LSTM','GRU','RNN'])
parser.add_argument('-drop','--dropout',default = 0.3,required=False,metavar="", type=float, help=' ',choices = [0.1,0.2,0.3,0.4,0.5])
parser.add_argument('-attn','--attentionRequired',default = True,required=False,metavar="", type=bool, help=' ',choices=[True,False])
parser.add_argument('-layer','--no_of_layers',default = 1,required=False,metavar="", type=int, help=' ')
args = parser.parse_args()

if torch.cuda.is_available():
    # If CUDA is available, use a CUDA device
    device = torch.device("cuda")
else:
    # If CUDA is not available, use the CPU
    device = torch.device("cpu")

traindata = pd.read_csv('/content/hin_train.csv',names= ['English','Hindi'],header = None)

testdata = pd.read_csv('/content/hin_test.csv',names = ['English','Hindi'],header = None)

valdata = pd.read_csv('/content/hin_valid.csv',names = ['English','Hindi'],header = None)


def tokenize(word):
    tokens = []
    for x in word:
        tokens.append(x)
    return tokens

max_eng_len = 0
max_hin_len = 0
test_max_eng_len = 0
test_max_hin_len = 0
val_max_eng_len = 0
val_max_hin_len = 0

for x in range(len(testdata)):
    temp = 0
    for y in testdata.iloc[x]['English']:
        temp+=1
    test_max_eng_len = max(test_max_eng_len,temp)

for x in range(len(testdata)):
    temp = 0
    for y in testdata.iloc[x]['Hindi']:
        temp +=1
    test_max_hin_len = max(test_max_hin_len,temp)

for x in range(len(valdata)):
    temp = 0
    for y in valdata.iloc[x]['English']:
        temp+=1
    val_max_eng_len = max(val_max_eng_len,temp)

for x in range(len(valdata)):
    temp = 0
    for y in valdata.iloc[x]['Hindi']:
        temp+=1
    val_max_hin_len = max(val_max_hin_len,temp)

English_vocab = []
for x in range(len(traindata)):
    temp = 0
    for y in traindata.iloc[x]['English']:
        temp += 1
        if y not in English_vocab:
            English_vocab.append(y)
    if(temp>max_eng_len):
        max_eng_len = max(max_eng_len,temp)

Hindi_vocab = []
for x in range(len(traindata)):
    temp = 0
    for y in traindata.iloc[x]['Hindi']:
        temp += 1
        if y not in Hindi_vocab:
            Hindi_vocab.append(y)
    max_hin_len = max(temp,max_hin_len)
for x in range(len(testdata)):
    for y in testdata.iloc[x]['Hindi']:
        if y not in Hindi_vocab:
            Hindi_vocab.append(y)

English_vocab = sorted(English_vocab)
Hindi_vocab = sorted(Hindi_vocab)

Eng_dict = {}
reverse_Eng = {}

for x in range(len(English_vocab)):
    Eng_dict[English_vocab[x]] = x+3
    reverse_Eng[x+3] = English_vocab[x]
Eng_dict['<sow>'] = 0
Eng_dict['<eow>'] = 1
Eng_dict['<pad>'] = 2
reverse_Eng[0] = '<sow>'
reverse_Eng[1] = '<eow>'
reverse_Eng[2] = '<pad>'


Hin_dict = {}
reverse_Hin = {}
for x in range(len(Hindi_vocab)):
    Hin_dict[Hindi_vocab[x]] = x+3
    reverse_Hin[x+3] = Hindi_vocab[x]
Hin_dict['<sow>'] = 0
Hin_dict['<eow>'] = 1
Hin_dict['<pad>'] = 2
reverse_Hin[0] = '<sow>'
reverse_Hin[1] = '<eow>'
reverse_Hin[2] = '<pad>'

def Eng_tokenize(word):
    tokens = []
    for x in word:
        tokens.append(Eng_dict[x])
    for x in range(len(tokens),max_eng_len):
        tokens.append(Eng_dict['<pad>'])
    return tokens

def Hin_tokenize(word):
    tokens = []
    for x in word:
        tokens.append(Hin_dict[x])
    tokens.append(Hin_dict['<eow>'])
    for x in range(len(tokens),max_hin_len+1):
        tokens.append(Hin_dict['<pad>'])
    return tokens

eng_word = []
hin_word = []
for x in range(len(traindata)):
    eng_word.append(Eng_tokenize(traindata.iloc[x]['English']))
    hin_word.append(Hin_tokenize(traindata.iloc[x]['Hindi']))

eng_word = torch.tensor(eng_word)
hin_word = torch.tensor(hin_word)

max_hin_len += 1
test_max_hin_len += 1
val_max_hin_len += 1

def test_Eng_tokenize(word):
    tokens = []
    for x in word:
        tokens.append(Eng_dict[x])
    for x in range(len(tokens),test_max_eng_len):
        tokens.append(Eng_dict['<pad>'])
    return tokens
def test_Hin_tokenize(word):
    tokens = []
    for x in word:
        tokens.append(Hin_dict[x])
    tokens.append(Hin_dict['<eow>'])
    for x in range(len(tokens),test_max_hin_len):
        tokens.append(Hin_dict['<pad>'])
    return tokens
def val_Eng_tokenize(word):
    tokens = []
    for x in word:
        tokens.append(Eng_dict[x])
    for x in range(len(tokens),val_max_eng_len):
        tokens.append(Eng_dict['<pad>'])
    return tokens
def val_Hin_tokenize(word):
    tokens = []
    for x in word:
        tokens.append(Hin_dict[x])
    tokens.append(Hin_dict['<eow>'])
    for x in range(len(tokens),val_max_hin_len):
        tokens.append(Hin_dict['<pad>'])
    return tokens
val_eng_word = []
val_hin_word = []
for x in range(len(valdata)):
    val_eng_word.append(val_Eng_tokenize(valdata.iloc[x]['English']))
    val_hin_word.append(val_Hin_tokenize(valdata.iloc[x]['Hindi']))
val_eng_word = torch.tensor(val_eng_word)
val_hin_word = torch.tensor(val_hin_word)
test_eng_word = []
test_hin_word = []
for x in range(len(testdata)):
    test_eng_word.append(test_Eng_tokenize(testdata.iloc[x]['English']))
    test_hin_word.append(test_Hin_tokenize(testdata.iloc[x]['Hindi']))
test_eng_word = torch.tensor(test_eng_word)
test_hin_word = torch.tensor(test_hin_word)

#------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self,char_embed_size,hidden_size,no_of_layers,dropout,rnn):
        super(Encoder,self).__init__()
        self.layer = no_of_layers
        self.rnn = rnn
        self.embedding = nn.Embedding(len(Eng_dict),char_embed_size).to(device)
        self.embedding.weight.requires_grad = True
        self.drop = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(char_embed_size,hidden_size,self.layer,batch_first = True,bidirectional = True).to(device)
        self.RNN = nn.RNN(char_embed_size,hidden_size,self.layer,batch_first = True,bidirectional = True).to(device)
        self.GRU = nn.GRU(char_embed_size,hidden_size,self.layer,batch_first = True,bidirectional = True).to(device)
    def forward(self,input,hidden,cell):
        embedded = self.embedding(input)
        embedded1 = self.drop(embedded)
        cell1 = cell
        if(self.rnn == 'RNN'):
            output,hidden1 = self.RNN(embedded1,hidden)
        elif(self.rnn == 'LSTM'):
            output,(hidden1,cell1) = self.LSTM(embedded1,(hidden,cell))
        elif(self.rnn == 'GRU'):
            output,hidden1 = self.GRU(embedded1,hidden)
        return output,(hidden1,cell1)


class Decoder(nn.Module):
    def __init__(self,char_embed_size,hidden_size,no_of_layers,dropout,batchsize,rnn):
        super(Decoder,self).__init__()
        self.layer = no_of_layers
        self.batchsize = batchsize
        self.hidden_size = hidden_size
        self.rnn = rnn
        self.embedding = nn.Embedding(len(Hin_dict),char_embed_size).to(device)
        self.drop = nn.Dropout(dropout)
        self.embedding.weight.requires_grad = True
        self.LSTM = nn.LSTM(char_embed_size + hidden_size*2,hidden_size,self.layer,batch_first = True).to(device)
        self.RNN = nn.RNN(char_embed_size + hidden_size*2,hidden_size,self.layer,batch_first = True).to(device)
        self.GRU = nn.GRU(char_embed_size + hidden_size*2,hidden_size,self.layer,batch_first = True).to(device)
        #2*hidden_size
        self.linear = nn.Linear(hidden_size,len(Hin_dict),bias=True).to(device)
        # dim = 2 
        self.softmax = nn.Softmax(dim = 2).to(device)
    def forward(self,input,hidden,cell,OGhidden,matrix):
        embedded = self.embedding(input)
        s1 = OGhidden.size()[1]
        s2 = OGhidden.size()[2]
        embedded1 = torch.cat((embedded,OGhidden[0].resize(s1,1,s2),OGhidden[1].resize(s1,1,s2)),dim = 2)
        embedded2 = self.drop(embedded1)
        cell1 = cell
        if(self.rnn == 'LSTM'):
            output,(hidden1,cell1) = self.LSTM(embedded2,(hidden,cell))
        elif(self.rnn == 'RNN'):
            output,hidden1 = self.RNN(embedded2,hidden)
        elif(self.rnn == 'GRU'):
            output,hidden1 = self.GRU(embedded2,hidden)
        output1 = self.linear(output)
        return output1,(hidden1,cell1)
        output2 = self.softmax(output1)
        return output2,hidden11
        
    #changed GRU char_embed_size
    #changed forward embedded

class Attention(nn.Module):
    def __init__(self,char_embed_size,hidden_size,no_of_layers,dropout,batchsize,rnn):
        super(Attention,self).__init__()
        self.layer = no_of_layers
        self.batchsize = batchsize
        self.hidden_size = hidden_size
        self.rnn = rnn
        self.embedding = nn.Embedding(len(Hin_dict),char_embed_size).to(device)
        self.drop = nn.Dropout(dropout)
        self.embedding.weight.requires_grad = True
        self.U = nn.Linear(hidden_size,hidden_size,bias = False).to(device)
        self.W = nn.Linear(hidden_size,hidden_size,bias = False).to(device)
        self.V = nn.Linear(hidden_size,1,bias = False).to(device)
        self.LSTM = nn.LSTM(char_embed_size + hidden_size,hidden_size,self.layer,batch_first = True).to(device)
        self.RNN = nn.RNN(char_embed_size + hidden_size,hidden_size,self.layer,batch_first = True).to(device)
        self.GRU = nn.GRU(char_embed_size + hidden_size,hidden_size,self.layer,batch_first = True).to(device) 
        self.linear = nn.Linear(hidden_size,len(Hin_dict),bias=True).to(device)
        self.softmax = nn.Softmax(dim = 2).to(device)
    def forward(self,input,hidden,cell,encoder_outputs,matrix):
        embedded = self.embedding(input)
        temp1 = self.U(encoder_outputs)
        temp2 = self.W(hidden[-1])
        s1 = temp2.size()[0]
        s2 = temp2.size()[1]
        add = temp1 + temp2.resize(s1,1,s2)
        tanh = F.tanh(add)
        ejt = self.V(tanh)
        ajt = nn.Softmax(dim = 1)(ejt)
        ct = torch.zeros(self.batchsize,1,self.hidden_size).to(device)
        ct = torch.bmm(ajt.transpose(1,2),encoder_outputs)
        final_input = torch.cat((embedded,ct),dim = 2)
        final_input = self.drop(final_input)
        cell1 = cell
        if(self.rnn == 'LSTM'):
            output,(hidden1,cell1) = self.LSTM(final_input,(hidden,cell))
        elif(self.rnn == 'RNN'):
            output,hidden1 = self.RNN(final_input,hidden)
        elif(self.rnn == 'GRU'):
            output,hidden1 = self.GRU(final_input,hidden)
        output1 = self.linear(output)
        if(matrix == True):
            return ajt,output1,(hidden1,cell1)
        return output1,(hidden1,cell1)


def Evaluate(attention,test_eng_word,test_hin_word,encoder,decoder,batchsize,hidden_size,char_embed_size,no_of_layers):
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        df = pd.DataFrame()
        en_hidden = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
        en_cell = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
        for x in range(0,len(testdata),batchsize):
            loss = 0
            input_tensor = test_eng_word[x:x+batchsize].to(device)
            if(input_tensor.size()[0] < batchsize):
                break
            output,(hidden,cell) = encoder.forward(input_tensor,en_hidden,en_cell)
            del(input_tensor)
            output = torch.split(output,[hidden_size,hidden_size],dim = 2)
            output = torch.add(output[0],output[1])/2
            input2 = []
            for y in range(batchsize):
                input2.append([0])
            input2 = torch.tensor(input2).to(device)
            hidden = hidden.resize(2,no_of_layers,batchsize,hidden_size)
            hidden1 = torch.add(hidden[0],hidden[1])/2
            cell = cell.resize(2,no_of_layers,batchsize,hidden_size)
            cell1 = torch.add(cell[0],cell[1])/2
            OGhidden = hidden1
            predicted = []
            predictions = []
            if(attention == True):
                temp = output
            else:
                temp = OGhidden
            for i in range(test_max_hin_len):
                output1,(hidden1,cell1) = decoder.forward(input2,hidden1,cell1,temp,False)
                predicted.append(output1)
                output2 = decoder.softmax(output1)
                output3 = torch.argmax(output2,dim = 2)
                predictions.append(output3)
                input2 = output3
            predicted = torch.cat(tuple(x for x in predicted),dim =1).to(device).resize(test_max_hin_len*batchsize,len(Hin_dict))
            predictions = torch.cat(tuple(x for x in predictions),dim =1).to(device)
            total_acc += accuracy(test_hin_word[x:x+batchsize].to(device),predictions,x)
            df = translate(test_hin_word[x:x+batchsize],predictions,df)
            loss  = nn.CrossEntropyLoss(reduction = 'sum')(predicted,test_hin_word[x:x+batchsize].reshape(-1).to(device))
            with torch.no_grad():
                total_loss += loss.item()
        test_loss = total_loss/(len(testdata)*test_max_hin_len)
        test_accuracy = (total_acc/len(testdata))*100
        del(predictions)
        del(predicted)
        del(input2)
        del(output1)
        del(output2)
        del(output3)
        del(hidden1)
        del(cell1)
        del(OGhidden)
        del(output)
        del(cell)
        return test_loss,test_accuracy,df

def valevaluate(attention,val_eng_word,val_hin_word,encoder,decoder,batchsize,hidden_size,char_embed_size,no_of_layers):
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for x in range(0,len(valdata),batchsize):
            loss = 0
            input_tensor = val_eng_word[x:x+batchsize].to(device)
#             en_hidden = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            if(input_tensor.size()[0] < batchsize):
                break
            en_hidden = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            en_cell = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            output,(hidden,cell) = encoder.forward(input_tensor,en_hidden,en_cell)
            del(input_tensor)
            del(en_hidden)
            del(en_cell)
            output = torch.split(output,[hidden_size,hidden_size],dim = 2)
            output = torch.add(output[0],output[1])/2
            input2 = []
            for y in range(batchsize):
                input2.append([0])
            input2 = torch.tensor(input2).to(device)
            hidden = hidden.resize(2,no_of_layers,batchsize,hidden_size)
            hidden1 = torch.add(hidden[0],hidden[1])/2
#             hidden1 = hidden[0]
            cell = cell.resize(2,no_of_layers,batchsize,hidden_size)
            cell1 = torch.add(cell[0],cell[1])/2
#             cell1 = cell[0]
            OGhidden = hidden1
            predicted = []
            predictions = []
            if(attention == True):
                temp = output
            else:
                temp = OGhidden
            for i in range(val_max_hin_len):
                output1,(hidden1,cell1) = decoder.forward(input2,hidden1,cell1,temp,False)
                predicted.append(output1)
                output2 = decoder.softmax(output1)
                output3 = torch.argmax(output2,dim = 2)
                predictions.append(output3)
                input2 = output3
            predicted = torch.cat(tuple(x for x in predicted),dim =1).to(device).resize(val_max_hin_len*batchsize,len(Hin_dict))
            predictions = torch.cat(tuple(x for x in predictions),dim =1).to(device)
            total_acc += accuracy(val_hin_word[x:x+batchsize].to(device),predictions,x)
            loss  = nn.CrossEntropyLoss(reduction = 'sum')(predicted,val_hin_word[x:x+batchsize].reshape(-1).to(device))
            with torch.no_grad():
                total_loss += loss.item()
#             print(loss.item())
        validation_loss = total_loss/(len(valdata)*val_max_hin_len)
        validation_accuracy = (total_acc/len(valdata))*100
        del(predictions)
        del(predicted)
        del(input2)
        del(output1)
        del(output2)
        del(output3)
        del(hidden1)
        del(cell1)
        del(OGhidden)
        del(output)
        del(cell)
        return validation_loss,validation_accuracy

def attentiontrain(batchsize,hidden_size,char_embed_size,no_of_layers,dropout,epochs,rnn):
    gc.collect()
    torch.autograd.set_detect_anomaly(True)
    encoder = Encoder(char_embed_size,hidden_size,no_of_layers,dropout,rnn).to(device)
    decoder = Attention(char_embed_size,hidden_size,no_of_layers,dropout,batchsize,rnn).to(device)
    print(encoder.parameters)
    print(decoder.parameters)
    opt_encoder = optim.Adam(encoder.parameters(),lr = 0.001)
    opt_decoder  = optim.Adam(decoder.parameters(),lr = 0.001)
    teacher_ratio = 0.5
    for _ in range(epochs):
        torch.cuda.empty_cache()
        print(_)
        total_loss = 0
        total_acc = 0
        for x in range(0,len(traindata),batchsize):
            loss = 0
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            input_tensor = eng_word[x:x+batchsize].to(device)
            en_hidden = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            en_cell = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            if(input_tensor.size()[0] < batchsize):
                break
            output,(hidden,cell) = encoder.forward(input_tensor,en_hidden,en_cell)
            output = torch.split(output,[hidden_size,hidden_size],dim = 2)
            output = torch.add(output[0],output[1])/2
            input2 = []
            for y in range(batchsize):
                input2.append([0])
            input2 = torch.tensor(input2).to(device)
            hidden = hidden.resize(2,no_of_layers,batchsize,hidden_size)
            hidden1 = torch.add(hidden[0],hidden[1])/2
            cell = cell.resize(2,no_of_layers,batchsize,hidden_size)
            cell1 = torch.add(cell[0],cell[1])/2
            predicted = []
            predictions = []
#             use_teacher_forcing = True if random.random() < teacher_ratio else False
            for i in range(max_hin_len):
                use_teacher_forcing = True if random.random() < teacher_ratio else False
                output1,(hidden1,cell1) = decoder.forward(input2,hidden1,cell1,output,False)
                predicted.append(output1)
                output2 = decoder.softmax(output1)
                output3 = torch.argmax(output2,dim = 2)
                predictions.append(output3)
                if(use_teacher_forcing):
                    input2 = hin_word[x:x+batchsize,i].to(device).resize(batchsize,1)
                else:
                    input2 = hin_word[x:x+batchsize,i].to(device).resize(batchsize,1)
            
            predicted = torch.cat(tuple(x for x in predicted),dim =1).to(device).resize(max_hin_len*batchsize,len(Hin_dict))
            predictions = torch.cat(tuple(x for x in predictions),dim =1).to(device)
            total_acc += accuracy(hin_word[x:x+batchsize].to(device),predictions,x)
            loss  = nn.CrossEntropyLoss(reduction = 'sum')(predicted,hin_word[x:x+batchsize].reshape(-1).to(device))
            with torch.no_grad():
                total_loss += loss.item()
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_norm = 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),max_norm = 1)
            opt_encoder.step()
            opt_decoder.step()
        del(input_tensor)
        del(en_hidden)
        del(en_cell)
        del(predictions)
        del(predicted)
        del(input2)
        del(output1)
        del(output2)
        del(output3)
        del(hidden)
        del(hidden1)
        del(cell1)
        del(output)
        del(cell)
        training_loss = total_loss/(51200*max_hin_len)
        training_accuracy = total_acc/512
        validation_loss,validation_accuracy = valevaluate(True,val_eng_word,val_hin_word,encoder,decoder,batchsize,hidden_size,char_embed_size,no_of_layers)
        print(f'Epoch = {_}')
        print(f'training_accuracy = {training_accuracy}')
        print(f'train_loss = {training_loss}')
        print('----------------------------------------')
        wandb.log({'training_accuracy' : training_accuracy, 'validation_accuracy' : validation_accuracy,'training_loss' : training_loss, 'validation_loss' : validation_loss,'epoch':_+1})
#         if(_ >= epochs/2):
#             teacher_ratio = 0
#         teacher_ratio /= 2
    return encoder,decoder


def train(batchsize,hidden_size,char_embed_size,no_of_layers,dropout,epochs,rnn):
    gc.collect()
    torch.autograd.set_detect_anomaly(True)
    encoder = Encoder(char_embed_size,hidden_size,no_of_layers,dropout,rnn).to(device)
    decoder = Decoder(char_embed_size,hidden_size,no_of_layers,dropout,batchsize,rnn).to(device)
    print(encoder.parameters)
    print(decoder.parameters)
    opt_encoder = optim.Adam(encoder.parameters(),lr = 0.001)
    opt_decoder  = optim.Adam(decoder.parameters(),lr = 0.001)
    teacher_ratio = 0.5
#     en_hidden = torch.randn(2*no_of_layers,batchsize,hidden_size).to(device)
    for _ in range(epochs):
        print(_)
        total_loss = 0
        total_acc = 0
        for x in range(0,len(traindata),batchsize):
            loss = 0
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            input_tensor = eng_word[x:x+batchsize].to(device)
            en_hidden = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            en_cell = torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            if(input_tensor.size()[0] < batchsize):
                break
            output,(hidden,cell) = encoder.forward(input_tensor,en_hidden,en_cell)
            del(en_hidden)
            del(en_cell)
            del(input_tensor)
            input2 = []
            for y in range(batchsize):
                input2.append([0])
            input2 = torch.tensor(input2).to(device)
            hidden = hidden.resize(2,no_of_layers,batchsize,hidden_size)
            cell = cell.resize(2,no_of_layers,batchsize,hidden_size)
            hidden1 = torch.add(hidden[0],hidden[1])/2
            cell1 = torch.add(cell[0],cell[1])/2
            OGhidden = hidden1
            predicted = []
            predictions = []
            use_teacher_forcing = True if random.random() < teacher_ratio else False
            if use_teacher_forcing:
                for i in range(max_hin_len):
                    output1,(hidden1,cell1) = decoder.forward(input2,hidden1,cell1,OGhidden,False)
                    predicted.append(output1)
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    predictions.append(output3)
                    input2 = hin_word[x:x+batchsize,i].to(device).resize(batchsize,1)
            else:
                for i in range(max_hin_len):
                    output1,(hidden1,cell1) = decoder.forward(input2,hidden1,cell1,OGhidden,False)
                    predicted.append(output1)
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    predictions.append(output3)
                    input2 = output3
            predicted = torch.cat(tuple(x for x in predicted),dim =1).to(device).resize(max_hin_len*batchsize,len(Hin_dict))
            predictions = torch.cat(tuple(x for x in predictions),dim =1).to(device)
            total_acc += accuracy(hin_word[x:x+batchsize].to(device),predictions,x)
#             print(predicted.shape)
#             print(hin_word[x:x+batchsize].reshape(-1).shape)
            loss  = nn.CrossEntropyLoss(reduction = 'sum')(predicted,hin_word[x:x+batchsize].reshape(-1).to(device))
            with torch.no_grad():
                total_loss += loss.item()
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_norm = 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),max_norm = 1)
            opt_encoder.step()
            opt_decoder.step()
        del(predictions)
        del(predicted)
        del(input2)
        del(output1)
        del(output2)
        del(output3)
        del(hidden1)
        del(cell1)
        del(OGhidden)
        del(output)
        del(cell)
        training_loss = total_loss/(51200*max_hin_len)
        training_accuracy = total_acc/512
        validation_loss,validation_accuracy = valevaluate(False,val_eng_word,val_hin_word,encoder,decoder,batchsize,hidden_size,char_embed_size,no_of_layers)
        wandb.log({'training_accuracy' : training_accuracy, 'validation_accuracy' : validation_accuracy,'training_loss' : training_loss, 'validation_loss' : validation_loss,'epoch':_+1})
#         if(_ >= epochs/2):
#             teacher_ratio = 0
    return encoder,decoder

def getword(characters):
    return "".join(characters)


def accuracy(target,predictions,flag):
    total = 0
    for x in range(len(target)):
        if(torch.equal(target[x],predictions[x])):
            total += 1
    return total


def translate(target,predictions,df):
    i = len(df)
    for x in range(len(predictions)):
        original = []
        for y in target[x]:
            if(y != 1):
                original.append(y)
            else:
                break
        predicted = []
        for y in predictions[x]:
            if(y != 1):
                predicted.append(y)
            else:
                break
        df.loc[i,['Original']] = getword([reverse_Hin[x.item()] for x in original])
        df.loc[i,['Predicted']] = getword([reverse_Hin[x.item()] for x in predicted])
        i+=1
    return df

batchsize = args.batchsize
hidden_size = args.hidden_size
char_embed_size = args.embedding_size
no_of_layers = args.no_of_layers
dropout = args.dropout
epochs = args.epochs
rnn = args.cell_type
wandb.login()
wandb.init(project= args.wandb_project,entity = args.wandb_entity)
if(args.attentionRequired == True):
    Encoder,Decoder = attentiontrain(batchsize, hidden_size, char_embed_size, no_of_layers, dropout, epochs, rnn)
else:
    Encoder,Decoder = train(batchsize, hidden_size, char_embed_size, no_of_layers, dropout, epochs, rnn)

test_loss,test_accuracy,predictions = Evaluate(True,test_eng_word,test_hin_word,Encoder,Decoder,batchsize,hidden_size,char_embed_size,no_of_layers)
print(test_loss,test_accuracy)