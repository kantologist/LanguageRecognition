import numpy as np
import pandas as pd
import os
import torch
from matplotlib import pyplot as plt
import time
import os
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
import librosa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

labels = np.load("labels.npy")
data = np.load("traindata.npy")
test_size = 6000
train_size = data.shape[0] - test_size
train = data[:train_size, :, :]
train_label = labels[:train_size]
test = data[train_size:, :, :]
test_label = labels[train_size:]

class TrainDataset(Dataset):
  
    def __init__(self, train, label):
        self.train = train
        self.label = label

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        train = torch.Tensor(self.train[idx])
        train = train.transpose(0,1)
        train = train.to(DEVICE)
        label = self.label[idx]
        label = label.astype(np.float)
#         print(label)
        label = torch.tensor(label)
        label = label.to(DEVICE)
        return(train, label)
    
class TestDataset(Dataset):
  
  def __init__(self, test):
    self.test = test
  
  def __len__(self):
    return self.test.shape[0]
  
  def __getitem__(self, idx):
    test = torch.Tensor(self.test[idx])
    test = test.to(DEVICE)
#     test = test.unsqueeze(0)
    return test

train_dataset = TrainDataset(train, train_label)
test_dataset = TestDataset(test)
train_batch_size = 5
test_batch_size = 5
train_loader = DataLoader(train_dataset,
                        batch_sampler=BatchSampler(RandomSampler(train_dataset), train_batch_size, False))
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

class DetectionModel(nn.Module):
    
    def __init__(self,vocab_size,embed_size,hidden_size, nlayers):
        super(DetectionModel,self).__init__()
        self.vocab_size=vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers=nlayers
        self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers, bidirectional=True, batch_first=True) # Recurrent network
        self.scoring = nn.Linear(hidden_size * 2,vocab_size) # Projection layer
        
    def forward(self, seq_batch):
        batch_size = seq_batch.size(0)
#         print(batch_size)
        embed = seq_batch #L x N x E
        hidden = None
        output_lstm,hidden = self.rnn(embed,hidden) #L x N x H
#         print(output_lstm.shape)
#         output_lstm_flatten = output_lstm.contiguous().view(-1,self.hidden_size * 2) #(L*N) x H
        output_flatten = self.scoring(output_lstm) #(L*N) x V
        return output_flatten
    
def train_epoch(model, optimizer, train_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = criterion.to(DEVICE)
    batch_id=0
    for inputs,targets in train_loader:
        batch_loss = []
        batch_id+=1
#         inputs = inputs.to(DEVICE)
#         targets = targets.to(DEVICE)
        outputs = model(inputs) # 3D
#         print(outputs.shape)
        outputs = outputs[:, -1, :] # pull out the last layer
#         print(outputs.shape)
#         print(targets.shape)
        loss = criterion(outputs,targets.long()) # Loss of the flattened outputs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        torch.cuda.empty_cache()
        if batch_id % 20 == 0:
            lpw = np.mean(batch_loss)
            print("At batch",batch_id)
            print("Training loss :",lpw)

    return model
    
langcount = 175
model = DetectionModel(langcount,40,256,3)
model = model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-6)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for i in range(50):
    train_epoch(model, optimizer, train_loader)