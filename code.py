# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

import csv

from collections import Counter

# Optimization module in scipy
from scipy import optimize

import math

from sklearn import metrics

import nltk
from nltk.stem.lancaster import LancasterStemmer



def readData(feature_size, stemming = False):
    
    # read the raw csv data
    reader = csv.reader(open("Data/dataASELECT.csv", "r"), delimiter=";")

    # make list of raw data
    rawList = list(reader)

    # convert data to numpy array
    dataArray = np.array(rawList)

    # vector containing the text messages
    messageList = dataArray[:,0]
    emotionList = dataArray[:,1]

    unique, counts = np.unique(emotionList, return_counts=True)
    
    # Visualizing the data
    pyplot.bar(unique,counts, color=['black', 'green', 'blue', 'yellow', 'orange', 'red'])
    pyplot.show()
    
    # Stemming
    if stemming:
        stemmer = LancasterStemmer()
    
        i = 0
        for message in messageList:
            mess = []
            w = nltk.word_tokenize(message)
            mess.extend(w)
            messageList[i] = ' '.join([stemmer.stem(w.lower()) for w in mess])
            i = i + 1



    # Split messages in training set (60%) -> because mostOcurringWords uses the training set (not full dataset)
    # total number of training examples
    m = len(emotionList);

    # 60% -> train set
    train_set_index = 0.6*m; # end index
           
    Xtrain = []
    for i in range (0, int(train_set_index)):
        Xtrain.append(messageList[i]);   
    

    
    # concatenate all messages into a String (delimiter is space)
    allText = ' '.join(Xtrain) #messageList.tolist()

    # for testing purposes
    from collections import Counter

    # list of every word in all the messages
    allTextSplit = allText.split()

    # counter object for selecting the most occuring words
    counterObject = Counter(allTextSplit)

    # list of words to ignore (these words are non-indicative of feeling)
    ignore = ['i']

    # list of most occuring words and number of occurences
    # note: len(ignore) extra elements, but these are filtered later
    mostOccuring = counterObject.most_common((len(ignore)) + feature_size) #should be e.g. 10000 (but for testing purposes 10)

    # get first element of all tuple (the words themselves)
    mostOccuringFirst = [tuple[0] for tuple in mostOccuring]

    # filter the "ignore" words out of mostOccuringFiltered and set all words to lowercase
    mostOccuringFiltered = [word.lower() for word in mostOccuringFirst if word not in ignore]


    X = []

    for message in messageList:

        trainingExample = []

        splittedMessage = message.split(' ') # should contain the splitted sentence

        splittedMessage = [word.lower() for word in splittedMessage] # set all words to lowercase

        for word in mostOccuringFiltered: 
            trainingExample.append(1) if word in splittedMessage else trainingExample.append(0)

        X.append(trainingExample)


    Y = []

    # get the different emotions in the training set
    emotions = np.unique(emotionList)

    for emotion in emotionList:

        #trainingExampleOutput = []

        comparisonBoolVector = emotion == emotions

        Y.append(1*comparisonBoolVector) # convert bool array to 0/1 array


    # convert X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    
    return messageList, mostOccuringFiltered, emotions, X, Y, m


def splitDataset(X, Y):
   
    # total number of training examples
    m = X.shape[0];

    # 60% -> train set
    train_set_index = 0.6*m; # end index
    #print(train_set_index)

    # 20% -> CV set
    cv_set_index = 0.2*m + train_set_index;
    #print(cv_set_index)

    # 20% -> test set
    test_set_index = 0.2*m + cv_set_index;
    #print(test_set_index)

    Xtrain = []
    Xcv = []
    Xtest = []
    Ytrain = []
    Ycv = []
    Ytest = []

    Xlist = list(X);
    Ylist = list(Y);

    for i in range (0, m):
        if (i <= train_set_index):
            Xtrain.append(Xlist[i]);
            Ytrain.append(Ylist[i]);
        elif (i > train_set_index and i < cv_set_index):
            Xcv.append(Xlist[i]);
            Ycv.append(Ylist[i]);
        else:
            Xtest.append(Xlist[i]);
            Ytest.append(Ylist[i]);

    Xtrain = np.array(Xtrain)
    Xcv = np.array(Xcv)
    Xtest = np.array(Xtest)
    Ytrain = np.array(Ytrain)
    Ycv = np.array(Ycv)
    Ytest = np.array(Ytest)

    
    return Xtrain, Xcv, Xtest, Ytrain, Ycv, Ytest




def sigmoidGradient(z):
    
    g = np.zeros(z.shape)

    g = sigmoid(z) * (1 - sigmoid(z))

    return g

def sigmoid(z):

    z = np.array(z)
    
    g = np.zeros(z.shape)

    g = 1 / (1 + np.exp(-z))

    return g

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):

    #epsilon_init = round(math.sqrt(6)/math.sqrt(L_in + L_out), 4)
    
    W = np.zeros((L_out, 1 + L_in))
    
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W
    
    

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_=0.0):
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = Y.shape[0]
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    # Forward propagation
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    a2 = sigmoid(a1.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    a3 = sigmoid(a2.dot(Theta2.T))
    
    temp1 = Theta1
    temp2 = Theta2
    
    # Add regularization term
    
    reg_term = (lambda_ / (2 * m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))
    
    
    a3[a3 == 0] = 10**(-10)
    a3[a3 == 1] = 1 - 10**(-10)

    
    J = (-1 / m) * np.sum((np.log(a3) * Y) + np.log(1 - a3) * (1 - Y)) + reg_term
    
    # Backpropogation
    
    delta_3 = a3 - Y
    delta_2 = delta_3.dot(Theta2)[:, 1:] * sigmoidGradient(a1.dot(Theta1.T))

    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)
    
    # Add regularization to gradient

    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    
    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    
    return J, grad


def trainNeuralNetwork(nnCostFunction, X, Y, lambda_, input_layer_size, hidden_layer_size, num_labels, iterations = 500):

    # Initializing Neural Network Parameters
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    
    # Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

    #  Number of iterations of the optimize function
    options= {'maxiter': iterations}


    # Create "short hand" for the cost function to be minimized
    costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                            hidden_layer_size,
                                            num_labels, X, Y, lambda_)

    #print('Before optimize')
    #print(np.array(Y).shape[0])

    # Now, costFunction is a function that takes in only one argument
    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)

    #print('After optimize')

    # get the solution of the optimization
    nn_params = res.x
    return nn_params



def predict(X, nn_params, input_layer_size, hidden_layer_size, num_labels):
    
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    
    
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    
    # Predict
    Xbiased = np.concatenate([np.ones((m, 1)), X], axis=1)
    a2 = sigmoid(Xbiased.dot(Theta1.T))
    
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    p = sigmoid(a2.dot(Theta2.T))#np.argmax(sigmoid(a2.dot(Theta2.T)), axis = 1)

    return p

def optimizer():

    #lambdas_to_test = [0, 0.01, 0.1, 1, 4, 6, 8, 10, 15] # dit zijn de verschillende 'lambdas'
    lambdas_to_test = [0, 0.01]
    
    #maxiter_to_test = [10, 100, 500, 1000] # dit zijn de verschillende 'maxiter'
    maxiter_to_test = [10, 50]
    
    #features_to_test
    features_to_test = [50, 100]
    
    #hidden_nodes_to_test
    hidden_nodes_to_test = [10, 20]
    
    #stemming
    stemming_to_test = [True, False]
    
    # Initialize variables
    err_array = []
    
    # Init these to the first value in the array
    lambd = 0
    itrat = 10
    feature = 50
    node = 10
    stemming = True

    # ====================== YOUR CODE HERE ======================
    for stem in stemming_to_test:
        
        for words in features_to_test:
            # Get the data (run as least as possible)
            messageList, mostOccuringFiltered, emotions, X, Y, m = readData(words, stem)
            Xtrain, Xcv, Xtest, Ytrain, Ycv, Ytest = splitDataset(X, Y)

            # Setup parameters for training the neural network
            input_layer_size  = len(mostOccuringFiltered) 
            num_labels = len(emotions)

            #print("-- nieuwe words")

            for lam in lambdas_to_test:

                #print ("-- nieuwe lambdas")

                for itr in maxiter_to_test:

                    #print ("-- nieuwe iteratie")

                    for size in hidden_nodes_to_test:

                        # Setup the parameters for training the neural network
                        hidden_layer_size = size

                        theta_t = trainNeuralNetwork(nnCostFunction, Xtrain, Ytrain, lam, input_layer_size, hidden_layer_size, num_labels, itr)

                        #predictions = predict(Xtrain, nn_params, input_layer_size, hidden_layer_size, num_labels)

                        cost, _ = nnCostFunction(theta_t, input_layer_size, hidden_layer_size, num_labels, Xcv, Ycv, lambda_ = 0)

                        if(not err_array or cost < min(err_array)): # if empty array or cost is smallest value
                            lambd = lam
                            itrat = itr
                            feature = words
                            node = size
                            stemming = stem
                        err_array.append(cost)
    
    # ============================================================
    return lambd, itrat, feature, node, stemming

print(optimizer())
