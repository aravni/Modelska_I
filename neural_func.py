# -*- coding: utf-8 -*-
'''Nevronske mreže 1.naloga: UČENJE'''

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
from tqdm import tqdm

# Add the directory containing your desired OpenMP library to PATH
os.environ['PATH'] = 'C:/Users/All Users/anaconda3/Library/bin/libiomp5md.dll;' + os.environ['PATH']
# Optionally, prevent conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, Hs):
        super(SimpleNN, self).__init__()
        self.flatten=nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28*28,Hs), # Input layer to hidden layer
            nn.ReLU(),
            nn.Linear(Hs,10), # Hidden layer to output layer
        )
    def forward(self, x):
        x = self.flatten(x) # Flatten the 28x28 images
        logits = self.stack(x)
        return logits
class Multilayer(nn.Module):
    def __init__(self, Nl):
        super(Multilayer, self).__init__()
        self.flatten = nn.Flatten()
        H = (28*28 - 10) // (Nl + 1)  # Hidden layer size based on Nl
        layers = []
        self.layer_sizes = [28*28] + [(Nl-i)*H for i in range(Nl)] + [10]  # layers sizes
        # Create the layers
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            layers.append(nn.Linear(input_size, output_size))
            if i < len(self.layer_sizes) - 2:  # No ReLU after output layer
                layers.append(nn.ReLU())
        self.stack = nn.Sequential(*layers)
    def forward(self, x):
        x = self.flatten(x) # Flatten the 28x28 images
        logits = self.stack(x)
        return logits
class ConvNN(nn.Module):
    def __init__(self, Hs, kernel_size):
        super(ConvNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=Hs, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=Hs, out_channels=2*Hs, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(Hs*2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        logits = self.stack(x)
        return logits

def learn(args):
    '''učenje mreže s 3 nivoji:
    Hs: size of hidden layer or number of hidden layers, 
    lr: parameter hitrosti učenja, 
    N_epochs: št. epoh za učenje - iteracije po vseh slikah.'''
    trainloader, testloader, Hs, lr, N_epoch = args
        
    # Instantiate the model, define the loss function and the optimizer
    # model = SimpleNN(Hs)
    # model = Multilayer(Hs) #Hs = num. of hidden layers
    model = ConvNN(Hs, kernel_size=3)

    loss_f = nn.CrossEntropyLoss(reduction='mean') #GPT predlaga - ne treba spreminjati dimenzij
    # loss_f = nn.L1Loss(reduction='mean') #mean absolute error
    # loss_f = nn.MSELoss(reduction='mean') #mean squared error / L2 norm

    optimizer = optim.SGD(model.parameters(), lr) #stochastic gradient descent - most basic
    # optimizer = optim.Adam(model.parameters(), lr) #Adam (Adaptive Moment Estimation)

    # Training loop
    correct=[]
    test_loss=[]
    t1=timer()
    # for epoch in tqdm(range(N_epoch), desc='epoch'):
    for epoch in range(N_epoch):
        for images, labels in trainloader:
            # Compute prediction error
            outputs = model(images)  # Forward pass
            # labels = F.one_hot(labels, num_classes=10).float()
            loss = loss_f(outputs, labels)  # Compute loss

            # Backpropagation
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights
            optimizer.zero_grad()  # Zero the parameter gradients

        c,l=test(testloader, model, loss_f)
        print(f"Epoch {epoch+1}/{N_epoch}: \t Accuracy: {(100*c):>0.1f}%, Avg loss (error from loss func): {l:>8f}")
        correct.append(c)
        test_loss.append(l)
    t2=timer()
    print(f"Finished Training. t= {(t2-t1):.1f} s")
    return model, loss_f, np.array(correct), np.array(test_loss), t2-t1

def test(dataloader, model, loss_f):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X) #prediction
            # y = F.one_hot(y, num_classes=10).float() #convert to right dimension vector
            test_loss += loss_f(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss

def bench(dataloader, model, loss_f):
    size = len(dataloader.dataset) #number of all samples in dataloader
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    calc_val=np.zeros(size,dtype=int) #calculated values
    correct = np.full(size, False) #True if calculated value is same as labeled (y)
    predicted = np.zeros((size,10)) #napovedane vrdnosti (rezultat zadnje plasti)
    i=0
    n_f=0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_f(pred, y).item()
            val=pred.argmax(1)
            batch_size=len(val)
            calc_val[i:i+batch_size] = val
            for j,v in enumerate(val):
                if (v==y[j]): correct[i+j] = True
                else: n_f +=1
                a=pred[j].numpy()
                predicted[i+j] = (a+abs(min(a)))/(abs(max(a))+abs(min(a)))
            i+=batch_size
    test_loss /= num_batches
    return test_loss, calc_val, correct, n_f, predicted

def deep_dream(model, loss_f, number, N_iter, lr, tol=1e-3):
    '''kateri vhod minimizira napako idealnega izhoda'''
    output=torch.zeros(1, 10) #ideal output
    output[0,number] = 1
    model.requires_grad = False #Disable gradients for the network - do not update model weights
    for p in model.parameters():
        p.requires_grad = False

    # Initialize a random image
    input = torch.zeros(1,1,28,28, requires_grad=True)

    optimizer = optim.SGD([input], lr) #stochastic gradient descent - most basic
    # for i in range(N_iter):  # Number of iterations
    i=0
    loss=1.
    while abs(loss) > tol and i < N_iter:
        # error of calculated values from input and desired output
        out=model(input)
        loss = loss_f(out, output)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        input.data = torch.clamp(input.data, 0, 1)
        if i<10: loss=1. #min 10 iter.
        i+=1
    print(f'number = {number}: final loss = {loss:.3E}, N_iter = {i}')
    return input.detach().cpu().numpy().reshape((28,28)), loss

def gradient(model, loss_f, number, input):
    '''kaj najbolj vpliva na odločitev'''
    output=np.full(10, 0)
    output[number] = 1
    output = torch.tensor(np.array([output]), dtype=torch.float32) #ideal output
    # input = torch.tensor(np.array([input]), dtype=torch.float32)
    input = torch.tensor(np.array([[input]]).reshape((1,1,28,28)), dtype=torch.float32) #konvolucijska
    input.requires_grad = True

    # error of calculated values from input and desired output
    out = model(torch.clamp(input,0,1))
    loss = loss_f(out, output)
    napoved = out.argmax(1).numpy()[0]
    #compute gradiant
    grad = torch.autograd.grad(loss, input, retain_graph=True)

    # Visualize the image at each step (optional)
    print(f'number = {number}: loss = {loss:.3f}')
    img = torch.clamp(grad[0][0],0,1).detach().numpy().reshape((28,28))
    out = torch.clamp(out,0,1).detach().numpy()
    return img, napoved, out, loss