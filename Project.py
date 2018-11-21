import numpy as np
import time
import torch.utils.data
from torch.autograd import Variable 
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR 
import pylab as plt
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from preprocessing import preprocess

labels = np.load("labels.npy")
data = np.load("traindata.npy")

test_size = 10000
train_size = data.shape[0] - test_size
print(train_size)
train = data[:train_size, :, :]
train_label = labels[:train_size]
test = data[train_size:, :, :]
test_label = labels[train_size:]


class customDataset(torch.utils.data.Dataset):
    """
    This is the custom Dataset to preprocess 
    the dataset
    """
    def __init__(self, data, label):
        super(customDataset, self).__init__()
        self.data = data
        self.label = label
        self.len = len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]
    def __len__(self):
        return self.len


def default_collate(batch):
    inputs,targets = zip(*batch)
    inputs = [preprocess(x) for x in inputs]
    inputs = torch.from_numpy(np.array(inputs)).float() 
    targets = torch.from_numpy(np.array(targets)).long()
    return inputs, targets  

class Block(nn.Module):
    def __init__(self,channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(channel, channel,3, 1,1, bias=False)
        self.conv2 = nn.Conv1d(channel, channel,3, 1,1, bias=False)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.elu(out)
        out = self.conv2(out)
        out += residual
        out = F.elu(out)
        return out

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.conv0 = nn.Conv1d(40, 64,3,1,1, bias=False)
        self.drop1 = nn.Dropout(0.2)
        self.same1 = Block(64)
        self.conv1 = nn.Conv1d(64,128,3,1,1, bias=False)
        self.drop2 = nn.Dropout(0.5)
        self.same2 = Block(128)
        self.conv2 = nn.Conv1d(128,256,3,1,1, bias=False)
        self.linear = nn.Linear(256,176, bias=False)
        for moudle in self.modules():
            if isinstance(moudle, nn.Conv2d):
                nn.init.kaiming_normal_(moudle.weight,mode="fan_out") 
            if isinstance(moudle, nn.Linear):
                nn.init.xavier_uniform_(moudle.weight.data)
            
    def forward(self,x):
        out = self.drop1(x)
        out = F.elu(self.conv0(out))
        out = self.drop2(out)
        out = self.same1(out)
        out = self.drop2(out)
        out = F.elu(self.conv1(out))
        out = self.drop2(out)
        out = self.same2(out)
        out = self.drop2(out)
        out = F.elu(self.conv2(out))
        out = torch.mean(out,2)
        out = self.linear(out)
        return out    



net = LanguageModel()
Traindataset = customDataset(train, train_label)
Devdataset = customDataset(test, test_label)

"""Trainning dataloader"""
train_loader= torch.utils.data.DataLoader(Traindataset, batch_size= 8, shuffle=True, collate_fn = default_collate)

"""validation dataloader"""
validate_loader = torch.utils.data.DataLoader(Devdataset, batch_size= 1, shuffle=False, collate_fn = default_collate)


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

criterion = nn.CrossEntropyLoss()



if torch.cuda.is_available():
    print('Model and parameters to cuda')
    net = net.cuda()
    criterion=criterion.cuda()
    
optimizer = torch.optim.Adam(net.parameters(),lr= 0.005)


"""Training our model"""
strt_train = time.time()
for epoch in range(100):
    losses = []
    myloss = []
    mytime = []
    net.train()
    for i, ( data, labels) in enumerate (train_loader):
            data = data.cuda()
            labels = labels.cuda()
            data = to_variable(data)
            optimizer.zero_grad()
            outputs = net(data)
            loss =criterion(outputs,torch.autograd.Variable(labels))
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            myloss.append(np.asscalar(np.mean(losses)))
            now = time.time() - strt_train 
            mytime.append(now)
            optimizer.step()
            if i%20==0:
                print (np.mean(losses))
                plt.title('The train losses for epoch {}'.format(epoch))
                plt.plot(mytime,myloss, 'r')
                plt.xlabel('Time')
                plt.ylabel('Error')
                plt.grid()
                plt.show()
                torch.save(net.state_dict(),'projectmodel.pt')



