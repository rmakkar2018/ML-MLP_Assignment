from Q1 import *
import os
import numpy as np
import pickle
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Loading the Training Dataset
X_train,Y_train=loadlocal_mnist(images_path='/home/rohit/Downloads/train-images-idx3-ubyte',labels_path='/home/rohit/Downloads/train-labels-idx1-ubyte')

# Loading the Testing Dataset
X_test,Y_test=loadlocal_mnist(images_path='/home/rohit/Downloads/t10k-images-idx3-ubyte',labels_path='/home/rohit/Downloads/t10k-labels-idx1-ubyte')
# normalising the data
X_train, X_test = X_train / 255.0, X_test / 255.0

###########################Relu actvation function
#getting weights
# file = open(os.path.join(os.getcwd(),("Weights/2_NN_1") ),'rb')
# print(file)
# nn1=pickle.load(file)
nn1=MyNeuralNetwork(5,[784,256,128,64,10],'relu', 0.1,'normal',100,100)
print("Training")
nn1.fit(X_train,Y_train)

## part a
print("Accuracy Relu : ", nn1.score(X_test, Y_test))

#### uncomment for part e
# x =nn1.y_after[3]
# y=Y_test
# tsne=TSNE(n_components=2)
# x1=tsne.fit_transform(x)
# sns.scatterplot(x=x1[:,0],y=x1[:,1],  hue=y)
# plt.show()
##### part e ends

###uncomment for part b plot
# nn1.plot(X_train,X_test, Y_train, Y_test )

#####storing model
# fpath = os.path.join(os.getcwd(),("Weights/2_NN_1") )
# file=open(fpath,'ab')
# pickle.dump(nn1,file)
# file.close()


################################tanh actvation function
# file = open(os.path.join(os.getcwd(),("Weights/2_NN_2") ),'rb')
# nn2=pickle.load(file)
nn2=MyNeuralNetwork(5,[784,256,128,64,10],'tanh', 0.1,'normal',100,100)
print("Training")
nn2.fit(X_train,Y_train)

## part a
print("Accuracy tanh : ", nn2.score(X_test, Y_test))


###uncomment for part b plot
# nn2.plot(X_train,X_test, Y_train, Y_test)


# ####storing the model
# fpath = os.path.join(os.getcwd(),("Weights/2_NN_2"))
# file=open(fpath,'ab')
# pickle.dump(nn2,file)
# file.close()


################################sigmoid actvation function
# file = open(os.path.join(os.getcwd(),("Weights/2_NN_3") ),'rb')
# nn3=pickle.load(file)
nn3=MyNeuralNetwork(5,[784,256,128,64,10],'sigmoid', 0.1,'normal',100,100)
print("Training")
nn3.fit(X_train,Y_train)

####part a
print("Accuracy sigmoid : ", nn3.score(X_test, Y_test))

###uncomment for part b plot
# nn3.plot(X_train,X_test, Y_train, Y_test )

#####storing model
# fpath = os.path.join(os.getcwd(),("Weights/2_NN_3") )
# file=open(fpath,'ab')
# pickle.dump(nn3,file)
# file.close()


###########################################linear actvation function
# file = open(os.path.join(os.getcwd(),("Weights/2_NN_4") ),'rb')
# nn4=pickle.load(file)
nn4=MyNeuralNetwork(5,[784,256,128,64,10],'linear', 0.1,'normal',100,100)
print("Training")
nn4.fit(X_train,Y_train)

# #part a
print("Accuracy linear : ", nn4.score(X_test, Y_test))

###uncomment for part b plot
# nn2.plot(X_train,X_test, Y_train, Y_test )

####saving model
# fpath = os.path.join(os.getcwd(),("Weights/2_NN_4") )
# file=open(fpath,'ab')
# pickle.dump(nn4,file)
# file.close()


############################# part f

clf=MLPClassifier(activation='relu',random_state=69, alpha=0.1, max_iter=100).fit(X_train,Y_train)
print("Relu accuracy: ",clf.score(X_test,Y_test))

clf=MLPClassifier(activation='identity',random_state=69, alpha=0.1, max_iter=100).fit(X_train,Y_train)
print("Linear accuracy: ",clf.score(X_test,Y_test))

clf=MLPClassifier(activation='tanh',random_state=69, alpha=0.1, max_iter=100).fit(X_train,Y_train)
print("Tanh accuracy: ",clf.score(X_test,Y_test))

clf=MLPClassifier(activation='logistic',random_state=69, alpha=0.1, max_iter=100).fit(X_train,Y_train)
print("Sigmoid accuracy: ",clf.score(X_test,Y_test))
