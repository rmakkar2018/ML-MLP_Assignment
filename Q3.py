import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim
import csv
import random
#https://stackoverflow.com/questions/60440292/runtimeerror-expected-scalar-type-long-but-found-float

###reading train and validation data 
f=open("/home/rohit/Downloads/largeTrain.csv",'r')
train_data=np.array(list(csv.reader(f,delimiter=',')),dtype=float)

f=open("/home/rohit/Downloads/largeValidation.csv",'r')
val_data=np.array(list(csv.reader(f,delimiter=',')),dtype=float)

####creating numpy arrays from 
X_train = train_data[:,1:]
Y_train = train_data[:,0]

X_val = val_data[:,1:]
Y_val = val_data[:,0]

def create_batches(X,Y,batch_size):
	#it returns the batches of batch_size of given X,Y 
	arr=np.arange(len(Y))
	batches=[]
	num_batches=X.shape[0]//batch_size
	for i in range(num_batches):
		start=i*batch_size
		batch=[]
		batch.append(torch.tensor(np.copy(X[arr[start:start+batch_size]])).float())
		batch.append(torch.tensor(np.copy(Y[arr[start:start+batch_size]])).long())
		batches.append(batch)
	return batches

##creating batches using training data
train_batches=create_batches(X_train,Y_train,36)

###converting into tensor arrays
X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).long()

X_val = torch.tensor(X_val).float()
Y_val = torch.tensor(Y_val).long()

def create_model(num_of_hidden_layers, learning_rate,num_epochs=100):

	model=nn.Sequential(nn.Linear(128,num_of_hidden_layers), nn.ReLU(),nn.Linear(num_of_hidden_layers,10),nn.LogSoftmax(dim=1))

	criterion =  nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	training_loss_itr=[]
	validation_loss_itr=[]

	for i in range(num_epochs):
		for batch in train_batches:
			x,y=batch[0],batch[1]

			optimizer.zero_grad()
			#forward propagation 
			out=model(x)
			#calculating the cross entropy loss
			loss=criterion(out,y)
			#back propagation
			loss.backward()
			#optimise
			optimizer.step()

		#calculating training loss and validation loss
		out = model(X_train)
		loss = criterion(out, Y_train)
		train_loss = loss.item()

		out = model(X_val)
		loss = criterion(out, Y_val)
		val_loss = loss.item()

		training_loss_itr.append(train_loss)
		validation_loss_itr.append(val_loss)

	return training_loss_itr[-1], validation_loss_itr[-1], training_loss_itr, validation_loss_itr



def plot_loss_of_hidden_layers(losses,hidden_layer_no=[4,5,20,50,100,200]):
	plt.plot(hidden_layer_no,losses[0],label='training loss')
	plt.plot(hidden_layer_no,losses[1],label='validation loss')
	plt.ylabel('Cross Entropy loss')
	plt.xlabel('Number of hidden layers')
	plt.legend()
	plt.show()

def plot_loss_vs_itr(training_loss_itr,validation_loss_itr):
	itr=np.arange(0,len(training_loss_itr))
	plt.plot(itr,training_loss_itr,label='training loss')
	plt.plot(itr,validation_loss_itr,label='validation loss')
	plt.ylabel('Cross Entropy loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()




#######part a
inputs=[[4,0.01],[5,0.01],[20,0.01],[50,0.01],[100,0.01],[200,0.01]]
outputs=[]


for i in range(len(inputs)):
	output=create_model(inputs[i][0],inputs[i][1])
	outputs.append(output)

hidden_layers_losses=[[],[]]
for i in range(len(outputs)):
	Cross_entropy_train_loss=outputs[i][0]
	Cross_entropy_val_loss=outputs[i][1]
	hidden_layers_losses[0].append(Cross_entropy_train_loss)
	hidden_layers_losses[1].append(Cross_entropy_val_loss)


plot_loss_of_hidden_layers(hidden_layers_losses)

#######part b

inputs=[[4,0.1],[4,0.01],[4,0.001]]
outputs=[]

for i in range(len(inputs)):
	output=create_model(inputs[i][0],inputs[i][1])
	outputs.append(output)

for i in range(len(outputs)):
	plot_loss_vs_itr(outputs[i][2],outputs[i][3])
