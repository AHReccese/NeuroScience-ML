#!/usr/bin/env python
# coding: utf-8

# # <font color='black'>EE25737: Introduction to Machine Learning</font>
# ## <font color='black'>Fall 99-00, Group 2</font>

# ### AmirHoseinRostami
# ### 96101635

# # C4: Neural Networks

# ## Requirements

# For installing Pytorch, run the following code: (if you don't use Jupter Notebook, first you should install pip command)

# In[ ]:


pip install torch


# Tensorflow:

# In[ ]:


pip install tensorflow


# For installing Keras, first you should install Tensorflow:

# In[ ]:


pip install keras


# In[16]:


# pre-process
# centering all graphs

from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# ## Load data

# In[2]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle

x_data = pd.read_csv("dataset/C4-Data/X_train.txt",sep=" ",header=None)
y_data = pd.read_csv("dataset/C4-Data/y_train.txt",sep=" ",header=None)

x_data['label'] = y_data
shuffledData = shuffle(x_data)
shuffledData = shuffledData.astype({shuffledData.columns[-1]: int})

length = len(shuffledData)
print(length)
# considering 10 percent of total data to be our test Set.
testSetSize = int(0.1*length) + 1
testSet = shuffledData[0:testSetSize]
trainSet = shuffledData[testSetSize:] 

cols = shuffledData.columns
x_train = trainSet[cols[0:-1]].to_numpy()
y_train = trainSet[cols[-1]].to_numpy()

x_test = testSet[cols[0:-1]].to_numpy()
y_test = testSet[cols[-1]].to_numpy()

#x_train.head()
## Code for loading the data


# ## C4.1.

# In[4]:


# using keras.
import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam

iterations = [10,100,200,300,400]
hiddenSize = 8

D_in = len(cols)-1
D_out = 12

# define the keras model
model = Sequential([
    
  Dense(hiddenSize, activation='relu', input_dim=D_in),
    
  Dense(hiddenSize, activation='relu'),
  Dense(hiddenSize, activation='relu'),
  Dense(hiddenSize, activation='relu'),
  Dense(hiddenSize, activation='relu'),
  Dense(hiddenSize, activation='relu'),
  Dense(hiddenSize, activation='relu'),
    
  Dense(D_out, activation='softmax')
])

# sgd = SGD(lr=0.01,momentum=0.9)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

train_loss = []
test_loss = []
NoOfClasses = 12

for iteration in iterations:
    
    model.fit(x_train, to_categorical(y_train-1),epochs=iteration)
    test_risk = 1 - model.evaluate(x_test, to_categorical(y_test-1))[1]
    train_risk = 1 - model.evaluate(x_train, to_categorical(y_train-1))[1]
    
    print("TestRisk: ",test_risk)
    print("TrainRisk: ",train_risk)
    
    test_loss.append(test_risk)
    train_loss.append(train_risk)
    
plt.plot(iterations, train_loss)
plt.xlabel('iteration') 
plt.ylabel('Loss') 
plt.title('EmpiricalLoss over iteration') 
plt.show() 

plt.plot(iterations, test_loss)
plt.xlabel('iteration')  
plt.ylabel('Loss') 
plt.title('TrueLoss over iteration') 
plt.show() 


# ## C4.2.

# In[5]:


## Part 2.

import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

iteration = 100 
hiddenSize = [1,2,4,8,16,32]
noOfhiddenLayers = 8

D_in = len(cols)-1
D_out = 12

def getHiddens(T,n):
    return [Dense(n, activation='relu') for i in range(T-1)]

train_loss = []
test_loss = []

for hidden in hiddenSize:
    # define the keras model
    model = Sequential([Dense(hidden, activation='relu', input_dim=D_in)] 
                       + getHiddens(noOfhiddenLayers,hidden) 
                       + [Dense(D_out, activation='softmax')])

    #sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

    model.fit(x_train, to_categorical(y_train-1),epochs=iteration)

    test_risk = 1 - model.evaluate(x_test, to_categorical(y_test-1))[1]
    train_risk = 1 - model.evaluate(x_train, to_categorical(y_train-1))[1]

    test_loss.append(test_risk)
    train_loss.append(train_risk)

plt.plot(hiddenSize, train_loss)
plt.xlabel('hiddenLayerDimension') 
plt.ylabel('Loss') 
plt.title('EmpiricalLoss over hiddenDimension') 
plt.show() 

plt.plot(hiddenSize, test_loss)
plt.xlabel('hiddenLayerDimension') 
plt.ylabel('Loss') 
plt.title('TrueLoss over hiddenDimension') 
plt.show() 


# ## C4.3.

# In[6]:


## Part 3.

import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

iteration = 100 
noOfhiddenLayers = [1,2,4,8,16]
hiddenSize = 8
D_in = len(cols)-1
D_out = 12

def getHiddens(T,n):
    return [Dense(n, activation='relu') for i in range(T-1)]

train_loss = []
test_loss = []

for number in noOfhiddenLayers:

    # define the keras model
    model = Sequential([Dense(hiddenSize, activation='relu', input_dim=D_in)] 
                       + getHiddens(number,hiddenSize) 
                       + [Dense(D_out, activation='softmax')])

    #sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

    model.fit(x_train, to_categorical(y_train-1),epochs=iteration) 

    test_risk = 1 - model.evaluate(x_test, to_categorical(y_test-1))[1]
    train_risk = 1 - model.evaluate(x_train, to_categorical(y_train-1))[1]

    test_loss.append(test_risk)
    train_loss.append(train_risk)

plt.plot(noOfhiddenLayers, train_loss)
plt.xlabel('hiddenLayerNum')  
plt.ylabel('Loss') 
plt.title('EmpiricalLoss over hiddenLayerNum') 
plt.show() 

plt.plot(noOfhiddenLayers, test_loss)
plt.xlabel('hiddenLayerNum')  
plt.ylabel('Loss') 
plt.title('TrueLoss over hiddenLayerNum') 
plt.show() 


# # C5. Multi-class Classification

# ## Load data

# In[15]:


## code
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv("dataset/fashion-mnist.csv",header=None)
data = data.drop(0)
data = data.astype({data.columns[-1]: str})
shuffledData = shuffle(data)

length = len(shuffledData)
print(length)
# considering 10 percent of total data to be our test Set.
testSetSize = int(0.5*length) + 1
testSet = shuffledData[0:testSetSize]
trainSet = shuffledData[testSetSize:] 

cols = shuffledData.columns
x_train = (trainSet[cols[0:-1]].to_numpy())/255
y_train = (trainSet[cols[-1]].to_numpy())

x_test = (testSet[cols[0:-1]].to_numpy())/255
y_test = (testSet[cols[-1]].to_numpy())

#x_train.head()
## Code for loading the data


# ## C5.1. SVM with linear kernel

# In[10]:


# forked from github https://github.com/DTrimarchi10.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


# In[11]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import seaborn as sns

# todo check linearSVC
svclassifier = LinearSVC(max_iter=100)
svclassifier.fit(x_train, y_train)

# calculate LinearSVC final errors
train_y_pred = svclassifier.predict(x_train) 
test_y_pred = svclassifier.predict(x_test)
cf_matrix = confusion_matrix(y_test, test_y_pred) 

#print(cf_matrix)
print(classification_report(y_test, test_y_pred))

make_confusion_matrix(cf_matrix, figsize=(8,6), cbar=False)


# ## C5.2. SVM with gaussian kernel

# In[12]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

bestGamma = 0.02
svclassifier = SVC(kernel='rbf',gamma= bestGamma)
svclassifier.fit(x_train, y_train)
print(svclassifier.gamma)

# calculate gaussianSVM final errors
train_y_pred = svclassifier.predict(x_train) 
test_y_pred = svclassifier.predict(x_test)

cf_matrix = confusion_matrix(y_test, test_y_pred) 

#print(cf_matrix)
print(classification_report(y_test, test_y_pred))

make_confusion_matrix(cf_matrix, figsize=(8,6), cbar=False)


# In[13]:


# extract the best gamma for gausian kernel
## code
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt  

gamas = [0.001,0.005,0.01,0.02,0.04,0.08,0.1,0.2,0.5,1]
error = []

for gama in gamas:
    
    print("Testing with gama = ",gama)
    svclassifier = SVC(kernel='rbf',gamma = gama)
    svclassifier.fit(x_train, y_train)
    
    # calculate gaussianSVM final errors
    train_y_pred = svclassifier.predict(x_train) 
    test_y_pred = svclassifier.predict(x_test)
    error.append(np.mean(test_y_pred != y_test))

plt.figure(figsize=(12, 6))
plt.plot(gamas, error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs gamma Value')
plt.xlabel('gamma Value')
plt.ylabel('Mean Error')


# ## C5.3. K-nearest neighbor

# In[14]:


## code
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(x_train, y_train)
test_y_pred = classifier.predict(x_test)
cf_matrix = confusion_matrix(y_test, test_y_pred) 

#print(cf_matrix)
print(classification_report(y_test, test_y_pred))

make_confusion_matrix(cf_matrix, figsize=(8,6), cbar=False)


# In[15]:


## code
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt  
    
# extract best K for knn
error = []
kmax = 10
# Calculating error for K values between 1 and 40
for i in range(1, kmax):
    print("Testing with K = ",i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, kmax), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# ## C5.4. Decision trees

# In[16]:


## code
## Train Desicion Tree for each depth here
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import classification_report, confusion_matrix

def calculateErrorOnSet(classifier,X,Y):
    predict = classifier.predict(X)
    error = 0
    for i in range(len(predict)):
        if(predict[i] != Y[i]):
            error = error + 1
    error = error / len(predict)
    return error
        
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
test_y_pred = clf.predict(x_test)

cf_matrix = confusion_matrix(y_test, test_y_pred) 
#print(cf_matrix)
print(classification_report(y_test, test_y_pred))
make_confusion_matrix(cf_matrix, figsize=(8,6), cbar=False)

validSetError = calculateErrorOnSet(clf,x_test,y_test)
print("validSetError is ",validSetError)


# ## C5.5. Neural network

# In[17]:


## code
# Phase1 testing.

import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import itertools

hiddenSize = 100

D_in = len(cols)-1 # 784
print(D_in)
D_out = 10

activeFuncs = ['relu','sigmoid','tanh']

def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 

subsets = findsubsets(set(range(3)),2)
phase1empLosses = [[] for _ in range(2*len(subsets))]
phase1trueLosses = [[] for _ in range(2*len(subsets))]
phase1labels = ["" for _ in range(2*len(subsets))]

counter = 0
for sub in subsets:
    i,j = sub
    for _ in range(2):
        i,j = j,i
        # define the keras model
        model = Sequential([
          Dense(hiddenSize, activation=activeFuncs[i], input_dim=D_in),    
          Dense(hiddenSize, activation=activeFuncs[j]),
          Dense(D_out, activation='softmax')
        ])

        empiricalRisks = []
        trueRisks = []
        iterations = [1]
        iterations.extend([i*10 for i in range(1,10)])
        for iteration in iterations:

            sgd = SGD(lr=0.01, momentum=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

            model.fit(x_train, to_categorical(y_train),epochs=iteration) 
            test_risk = 1 - model.evaluate(x_test, to_categorical(y_test))[1]
            train_risk = 1 - model.evaluate(x_train, to_categorical(y_train))[1]

            empiricalRisks.append(train_risk)
            trueRisks.append(test_risk)

        
        print("activation functions are: " + activeFuncs[i] + ", " + activeFuncs[j])
        print("Empirical Risks are:",empiricalRisks)
        print("True Risks are: ",trueRisks )
        
        phase1empLosses[counter] = empiricalRisks
        phase1trueLosses[counter] = trueRisks
        phase1labels[counter] = activeFuncs[i] + ", " + activeFuncs[j]
        counter = counter + 1
        


# In[18]:


# comparsion All togather

#phase1empLosses
#phase1trueLosses
#phase1labels

iterations = [1]
iterations.extend([i*10 for i in range(1,10)])

plt.figure(figsize=(12, 6))
for i in range(2*len(subsets)):    
    plt.plot(iterations, phase1empLosses[i],
             linestyle='dashed', marker='o',markersize=10)

plt.title('Empirical Loss among different Activation functions')
plt.xlabel('IterationStep')
plt.ylabel('Empirical Mean Error')
plt.legend(phase1labels) 
plt.show()

plt.figure(figsize=(12, 6))
for i in range(2*len(subsets)):    
    plt.plot(iterations, phase1trueLosses[i],
             linestyle='dashed', marker='o',markersize=10)

plt.title('TrueLoss among different Activation functions')
plt.xlabel('IterationStep')
plt.ylabel('TrueLoss Mean Error')
plt.legend(phase1labels) 
plt.show()


# In[19]:


## code
# Phase2 testing.

import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import itertools

hiddenSize = 100

D_in = len(cols)-1 # 784
print(D_in)
D_out = 10


activeFuncs = ['relu','sigmoid','tanh']

phase2empLosses = [[] for _ in range(len(activeFuncs))]
phase2trueLosses = [[] for _ in range(len(activeFuncs))]
phase2labels = ["" for _ in range(len(activeFuncs))]
counter = 0

for func in activeFuncs:
    # define the keras model
    model = Sequential([
      Dense(hiddenSize, activation=func, input_dim=D_in),    
      Dense(hiddenSize, activation=func),
      Dense(D_out, activation='softmax')
    ])

    empiricalRisks = []
    trueRisks = []
    iterations = [1]
    iterations.extend([i*10 for i in range(1,10)])
    for iteration in iterations:

        sgd = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        model.fit(x_train, to_categorical(y_train),epochs=iteration) 
        test_risk = 1 - model.evaluate(x_test, to_categorical(y_test))[1]
        train_risk = 1 - model.evaluate(x_train, to_categorical(y_train))[1]

        empiricalRisks.append(train_risk)
        trueRisks.append(test_risk)

    print("activation functions are: " + func + ", " + func)
    print("Empirical Risks are:",empiricalRisks)
    print("True Risks are: ",trueRisks )
    
    phase2empLosses[counter] = empiricalRisks
    phase2trueLosses[counter] = trueRisks
    phase2labels[counter] = "both are " + func
    counter = counter + 1


# In[20]:


#phase2empLosses
#phase2trueLosses
#phase2labels

iterations = [1]
iterations.extend([i*10 for i in range(1,10)])

plt.figure(figsize=(12, 6))
for i in range(len(activeFuncs)):    
    plt.plot(iterations, phase2empLosses[i],
             linestyle='dashed', marker='o',markersize=10)

plt.title('Empirical Loss among different Activation functions')
plt.xlabel('IterationStep')
plt.ylabel('Empirical Mean Error')
plt.legend(phase2labels) 
plt.show()

plt.figure(figsize=(12, 6))
for i in range(len(activeFuncs)):    
    plt.plot(iterations, phase2trueLosses[i],
             linestyle='dashed', marker='o',markersize=10)

plt.title('TrueLoss among different Activation functions')
plt.xlabel('IterationStep')
plt.ylabel('TrueLoss Mean Error')
plt.legend(phase2labels) 
plt.show()


# In[21]:


## code
## code
# Phase3 testing.

import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import itertools

hiddenSize = 100

D_in = len(cols)-1 # 784
print(D_in)
D_out = 10

# define the keras model
model = Sequential([
  Dense(hiddenSize,input_dim=D_in),    
  Dense(hiddenSize),
  Dense(D_out, activation='softmax')
])

empiricalRisks = []
trueRisks = []

iterations = [1]
iterations.extend([i*10 for i in range(1,10)])

for iteration in iterations:

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    model.fit(x_train, to_categorical(y_train),epochs=iteration) 
    test_risk = 1 - model.evaluate(x_test, to_categorical(y_test))[1]
    train_risk = 1 - model.evaluate(x_train, to_categorical(y_train))[1]

    empiricalRisks.append(train_risk)
    trueRisks.append(test_risk)

print("activation functions both are linear")
print("Empirical Risks are:",empiricalRisks)
print("True Risks are: ",trueRisks )


plt.figure(figsize=(12, 6))
plt.plot(iterations, empiricalRisks,
         linestyle='dashed', marker='o',markersize=10)

plt.title('EmpiricalLoss(Linear Hidden Activation functions)')
plt.xlabel('IterationStep')
plt.ylabel('Empirical Mean Error')
plt.legend(["Both Linear"]) 
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(iterations, trueRisks,
         linestyle='dashed', marker='o',markersize=10)

plt.title('TrueLoss(Linear Hidden Activation functions)')
plt.xlabel('IterationStep')
plt.ylabel('TrueLoss Mean Error')
plt.legend(["both Linear"]) 
plt.show()


# In[18]:


## code
## bestCase -> todo

import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import itertools

hiddenSize = 100

D_in = len(cols)-1 # 784
print(D_in)
D_out = 10

# define the best NN model
model = Sequential([
  Dense(hiddenSize, activation='relu', input_dim=D_in),    
  Dense(hiddenSize, activation='relu'),
  Dense(D_out, activation='softmax')
])

empiricalRisks = []
trueRisks = []
iterations = [1]
iterations.extend([i*10 for i in range(1,10)])
for iteration in iterations:

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    model.fit(x_train, to_categorical(y_train),epochs=iteration) 
    test_risk = 1 - model.evaluate(x_test, to_categorical(y_test))[1]
    train_risk = 1 - model.evaluate(x_train, to_categorical(y_train))[1]

    empiricalRisks.append(train_risk)
    trueRisks.append(test_risk)

    print("Empirical Risks are:",empiricalRisks)
    print("True Risks are: ",trueRisks )


plt.figure(figsize=(12, 6))
plt.plot(iterations, empiricalRisks, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)

plt.title('TrainLoss per iteration')
plt.xlabel('IterationStep')
plt.ylabel('Mean Error')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(iterations, trueRisks, color='green', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)

plt.title('TrueLoss per iteration')
plt.xlabel('IterationStep')
plt.ylabel('Mean Error')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(iterations, empiricalRisks,
         iterations, trueRisks,
         linestyle='dashed', marker='o',markersize=10)

plt.title('Loss per iteration')
plt.xlabel('IterationStep')
plt.ylabel('Mean Error')
plt.legend(['EmpiricalRisk','TrueRisk']) 
plt.show()


# # C6. K-means

# ## Load data

# ## C6.1. Algorithm

# In[1]:


import random as rd

def k_means(matrix, K, max_iterations=100):
    
    n = matrix.shape[0] #number of training examples
    m = matrix.shape[1] #number of features. Here n=2
    
    Centroids=np.array([]).reshape(m,0)
    for i in range(K):
        rand=rd.randint(0,n-1)
        Centroids=np.c_[Centroids,matrix[rand]]

    for i in range(max_iterations):
        
        EuclidianDistance=np.array([]).reshape(n,0)
        for k in range(K):
            tempDist=np.sum((matrix-Centroids[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1)+1
        
        
        Y={}
        for k in range(K):
            Y[k+1]=np.array([]).reshape(m,0)
        for i in range(n):
            Y[C[i]]=np.c_[Y[C[i]],matrix[i]]
     
        for k in range(K):
            Y[k+1]=Y[k+1].T
    
        for k in range(K):
            Centroids[:,k]=np.mean(Y[k+1],axis=0)
        
        output=Y,Centroids
        
    return output


# In[2]:


## code
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv("dataset/iris.csv",header=None)
# cols: sepal.length  sepal.width  petal.length  petal.width  variety
data = data.drop(0)
for i in range(len(data.columns)-1):
    data = data.astype({data.columns[i]: float})
shuffledData = shuffle(data)


length = len(shuffledData)
print(length)
# considering 10 percent of total data to be our test Set.
testSetSize = 0
testSet = shuffledData[0:testSetSize]
trainSet = shuffledData[testSetSize:] 

cols = shuffledData.columns
x_train = trainSet[cols[0:-1]].to_numpy()
y_train = trainSet[cols[-1]].to_numpy()

x_test = testSet[cols[0:-1]].to_numpy()
y_test = testSet[cols[-1]].to_numpy()


# ## C6.2. Plot clusters in 2d

# In[5]:


import matplotlib.pyplot as plt
import itertools 
  
K = 4
kmeans_result,centroids = k_means(x_train,K,100)

def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 

subsets = findsubsets(set(range(x_train.shape[1])),2)
for sub in subsets:

    i,j = sub
    i,j = min(i,j),max(i,j)
    plt.scatter(x_train[:,i],x_train[:,j])
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':100})
    plt.title('Original Dataset')
    plt.xlabel('$X_{}$'.format(i+1),fontsize=18)
    plt.ylabel('$X_{}$'.format(j+1),fontsize=18)
    
    for k in range(1,K+1):
        plt.scatter(kmeans_result[k][:,i],kmeans_result[k][:,j])
    plt.scatter(centroids[i,:],centroids[j,:],s=100,c='yellow')
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':100})
    plt.show()
    


# ## C6.3. Redundant feature

# In[12]:


## code
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def removeColIndex(index,cols):
    result = []
    for item in cols[0:-1]:
        if item != index:
            result.append(item)
    return result

data = pd.read_csv("dataset/iris.csv",header=None)
# cols: sepal.length  sepal.width  petal.length  petal.width  variety
data = data.drop(0)
for i in range(len(data.columns)-1):
    data = data.astype({data.columns[i]: float})
shuffledData = shuffle(data)

length = len(shuffledData)
print(length)
# considering 10 percent of total data to be our test Set.
testSetSize = 0
testSet = shuffledData[0:testSetSize]
trainSet = shuffledData[testSetSize:] 

cols = shuffledData.columns
remove = 2 # X3 is good to remove
x_train = trainSet[removeColIndex(remove,cols)].to_numpy()
y_train = trainSet[cols[-1]].to_numpy()

x_test = testSet[removeColIndex(remove,cols)].to_numpy()
y_test = testSet[cols[-1]].to_numpy()

## Code for loading the data


# In[13]:


import matplotlib.pyplot as plt
import itertools 
  
K = 4
kmeans_result,centroids = k_means(x_train,K,100)

def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 

subsets = findsubsets(set(range(x_train.shape[1])),2)
for sub in subsets:

    i,j = sub
    i,j = min(i,j),max(i,j)
    plt.scatter(x_train[:,i],x_train[:,j])
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':100})
    plt.title('Original Dataset')
    iprim = i
    jprim = j
    
    if(iprim == remove):
        iprim = iprim + 1
        
    if(jprim == remove):
        jprim = jprim + 1
        
    
    plt.xlabel('$X_{}$'.format(iprim+1),fontsize=18)
    plt.ylabel('$X_{}$'.format(jprim+1),fontsize=18)
    
    for k in range(1,K+1):
        plt.scatter(kmeans_result[k][:,i],kmeans_result[k][:,j])
    plt.scatter(centroids[i,:],centroids[j,:],s=100,c='yellow')
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':100})
    plt.show()
    

