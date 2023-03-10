import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000
num_epoch = 500
x= init.uniform_(torch.Tensor(num_data,1),-10,10)
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)
y= 2*x +3
y_noise = y + noise
model = nn.Linear(1,1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

label = y_noise
for i in range(num_epoch) :
    optimizer.zero_grad()
    output =model(x)

print("hello world")
#loss_func = nn.L1Loss(y-y_noise)j


##### dkssudgktpwy ... ...
num_data=500
num_epoch = 1000
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)
x = init.uniform_(torch.FloatTensor(num_data ,1),-15,15)
y = (x**2) + 3
y_noise = y + noise

model = nn.Sequential(nn.Linear(1,6),
                      nn.ReLU(),
                      nn.Linear(6,10),
                      nn.ReLU(),
                      nn.Linear(10,6),
                      nn.ReLU(),
                      nn.Linear(6,1),)

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr=0.0002)

