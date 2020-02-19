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
        if intent['tag'] not in classes:#(in loop or not)
            #print(intent['tag'])
            classes.append(intent['tag'])
#print(words)
#print(documents)
print(classes)

#-----------lemmatizing, lowering each word and removing duplicates---------------------------
#print(words)
#print()
words=[(lemmatizer.lemmatize(word.lower())) for word in words if word not in ignoreWords]
#print(words)
#print(len(words))#185
words=sorted(list(set(words)))
#print(len(words))#87
classes=sorted(list(set(classes)))
#print(classes)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#-------------training and testing data----------------------------------------------------

#input will be the pattern and output is the class to which the input pattern belongs to
#print(documents)
training=[]
output=[0]*len(classes)
#print(output)
#print(documents)
#print(words)
#print(classes)
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
    #print(len(bag))
    outputRow=list(output)
    outputRow[classes.index(doc[1])]=1
    training.append([bag,outputRow])
    #print(training)
    #print()
    #print()
random.shuffle(training)
training=np.array(training)
#print(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
print(train_y)
#print(len(train_y))
print("Training Data Created")

#-----------------------------Model Building---------------------------------------------


model=Sequential()#linear stack of layers
model.add(Dense(128,input_shape=(len(train_x[0]),),activation="relu"))#either add directly in constructor
#dense layer req-num of layers, model needs to know what input shape the data is in, it accepts the tuple integration
#model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation="softmax"))
print(model.summary())
