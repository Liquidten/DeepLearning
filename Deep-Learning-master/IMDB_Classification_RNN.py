#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:46:45 2019

@author: sameepshah
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

print(tf.__version__)


'''
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc
'''
def splitdata(imdb):
    #num_words = 10000 keeps the 10,000 most frequent occured words from the training data 
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    return (train_data, train_labels), (test_data, test_labels)

#function to decode to text
def text_review(text):
    return ' '.join([text_data.get(i, '?') for i in text])

def data_padding(train_data,test_data):
    #Data Preperation, padding the array so all of the data has the same length
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding = 'post',
                                                            maxlen = 300)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding = 'post',
                                                           maxlen = 300)
    return (train_data,test_data)
    

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    
def ROC_plot(X_values_test,y_values_test,model2):
    # ROC Plot1
    plt.figure(0)
    
    #results = model2.predict_classes(np.array(test_data), batch_size = 1)

    CM_pred_proba = model2.predict_proba(X_values_test)[::,0]
    fpr, tpr, _ = metrics.roc_curve(y_values_test,  CM_pred_proba)
    auc = metrics.roc_auc_score(y_values_test, CM_pred_proba)
    plt.plot(fpr,tpr,label="data CNN, auc="+str(auc))
    plt.legend(loc=4)
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.title('ROC Plot')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
def RNNModel(train_data,train_labels):
    # Input the value, whether you want to run the model on LSTM RNN or GRU RNN.
    print("Input 'LSTM' for LSTM RNN, 'GRU' for GRU RNN ")
    modelInput= input("Do you want to compile the model using LSTM RNN or GRU RNN?\n")
    if modelInput == "LSTM":
        lstm = True
    else:
        lstm = False

    #Building a 1D Convvolutional Neural Network Model

    #input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000
    maxlen = 300
    embedding_vector_length = 32


    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_vector_length, input_length=maxlen))
    #model.add(keras.layers.Dropout(0.2))
    #LSTMmodel.add(keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    #model.add(keras.layers.MaxPool1D(pool_size = 2))
    if lstm == True:
        model.add(keras.layers.LSTM(150))
    else:
        model.add(keras.layers.GRU(150))
    
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    
    Adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = Adam,loss='binary_crossentropy', metrics=['acc', auc])

    #splitting the data for validation purposes 
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    pratial_y_train = train_labels[10000:]
    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                                               #verbose=0, mode='auto', baseline=None, 
                                               #restore_best_weights=False)
    history = model.fit(np.array(partial_x_train),np.array(pratial_y_train),epochs=40, 
                        batch_size=512, 
                        validation_data=(np.array(x_val),np.array(y_val)),
                        verbose=1)
    return model, history
    
def saveModel(model):
    # Saving Model for future API
    model.save('IMDB_Classification_RNN.h5')
    print("Saved model to disk")
    del model #deletes the existing model

def loadModel(des):
    #returns a compiled model
    #identical to the previous one
    model2 = keras.models.load_model('IMDB_Classification_RNN.h5', custom_objects={'auc': auc})
    return model2



def lossAccuracyPlot(history):
    history_dict = history.history
    history_dict.keys()
    
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__=="__main__":
    #importing imdb data from keras datasert 
    imdb = keras.datasets.imdb
    print(imdb)
    
    (train_data, train_labels), (test_data, test_labels) = splitdata(imdb)
    
    #Summarize number of unique words
    print("Number of words: ")
    print(len(np.unique(np.hstack(train_data))))
    
    #Data Exploration 
    print("Training entries: {}, labels".format(len(train_data), len(train_labels)))

    #Texts of reviews have been converted to integers, 
    #where integer represents a specific word in the dictionary
    print(train_data[4])

    #showing the reviews of varing length, below shows the num of words for 1st and 2nd review
    print(len(train_data[4]), len(train_data[5]))

    #lets view the average review length
    print("Review length: ")
    reviews = [len(x) for x in train_data]
    print("Mean %.2f words Standard Deviation (%f)" % (np.mean(reviews), np.std(reviews)))

    #plot review lenght
    pyplot.boxplot(reviews)
    pyplot.show()
    
    
    #Helper function to convert integers back to words

    #A dictonary mapping words to an integer index
    word_index = imdb.get_word_index()

    #The first indices are reserved
    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2 #unknown
    word_index["<UNUSED>"] = 3

    text_data = dict([(value, key) for (key, value) in word_index.items()])

    
    #using the text_review function to display the text for the 4th review:
    print(text_review(train_data[4]))
    
    train_data,test_data = data_padding(train_data,test_data)
    
    #checking to make sure the length are equal
    print(len(train_data[4]), len(train_data[5]))

    #inspecting one of the reviews
    print(train_data[4])

    
    model, history = RNNModel(train_data,train_labels)
    saveModel(model)
    
    des = "../Algo_Project/IMDB_Classification_CNN.h5"
    
    model2 = loadModel(des)
    results = model2.predict_classes(np.array(test_data), batch_size = 1)
    score = accuracy_score(np.array(test_labels), results)
    results2 = model2.evaluate(np.array(test_data), test_labels)
    print('Test accuracy:', score)
    print('Test accuracy:', results2)
    print(confusion_matrix(test_labels, results))
    ROC_plot(test_data,test_labels,model2)
    lossAccuracyPlot(history)



