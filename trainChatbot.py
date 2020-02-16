#------------------------Import and Load data file-----------------------------------------

#import tensorflow  #Required For Keras, as it uses tensorflow backend
#Ignoring warnings for Tensorlow alpha version (For Py3.7)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#Natural Language toolkit
import nltk
from nltk.stem import WordNetLemmatizer  #Lemmatizer
from nltk.tokenize import word_tokenize #Tokenizer

lemmatizer=WordNetLemmatizer()

import json #for serializing/Deserializing Python Objects
import pickle #To store serialzed object in Pickle file
import numpy as np
#Keras for building model for training the chatbot
from keras.models import Sequential  #Sequential Model Used
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import random

words=[]
classes=[]
documents=[]
ignoreWords=['!','?',',',"'",'.','-']
try:
    with open("intents.json","r") as dataFile:
        intents=json.load(dataFile)
except ValueError:
    print('Decoding JSON Failed!!!')
#print(type(intents)) #JSON to dict

#------------------Preprocess data------------------------------------------------------

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=word_tokenize(pattern)#tokenize each word
        words.extend(w)#storing each word in pattern
        documents.append((w,intent['tag'])) #add category corresponding to each patterns
        #adding all the categories in classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(words)
#print(documents)
#print(classes)

#-----------lemmatizing, lowering each word and removing duplicates---------------------------
#print(words)
#print()
words=[(lemmatizer.lemmatize(word.lower())) for word in words if word not in ignoreWords]
#print(words)
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#-------------training and testing data----------------------------------------------------

#input will be the pattern and output is the class to which the input pattern belongs to
#print(documents)
training=[]
output=[0]*len(classes)
for doc in documents:
    bag=[]
    patternWords=doc[0]
    #print(patternWords)
    patternWords=[lemmatizer.lemmatize(word.lower()) for word in patternWords]
    for w in words:
        if w in patternWords:
            bag.append(1)
        else:
            bag.append(0)
    outputRow=list(output)
    outputRow[classes.index(doc[1])]=1
    training.append([bag,outputRow])
random.shuffle(training)
training=np.array(training)
#print(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
print(train_y)
print(len(train_y))
print("Training Data Created")
