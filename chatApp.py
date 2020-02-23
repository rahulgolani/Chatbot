#--------------Predict the result based on the response of the user----------------------

#Model only tells what class the user input belongs to

#Ignoring warnings for Tensorlow alpha version (For Py3.7)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
try:
    with open("intents.json","r") as dataFile:
        intents=json.load(dataFile)
except ValueError:
    print('Decoding JSON Failed!!!')
words=pickle.load(open("words.pkl","rb"))
classes=pickle.load(open("classes.pkl","rb"))

#print(intents,end=" ")
#print(words,end=" ")
#print(classes,end=" ")

model=load_model('chatbot_model.h5')

def preprocessSentence(sentence):
    sentence_words=word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bow(sentence,words,showDetails=True):
    sentence_words=preprocessSentence(sentence)
    #bag=[]
    bag=[0]*len(words)
    #ind=words.index('good')
    #print(sentence_words)
    #print(words)
    #print(ind)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1

    '''
    for w in words:
        if w in sentence_words:
            bag.append(1)
            if showDetails:
                print("found in bag",w)
        else:
            bag.append(0)
    '''
    #print(bag[30])
    #print(bag)
    #print(len(bag))
    return np.array(bag)

def predictClass(sentence):
    p=bow(sentence,words,showDetails=False)
    #print(len(p))
    #p=np.array([p])
    #print(p)
    result=model.predict_classes(np.array([p]),batch_size=10,verbose=0)
    print(result[0])
    print(classes[result[0]])#gives result as greeting
    resultClass=classes[result[0]]
    return resultClass


def chatbot_response(text):
    resultClass=predictClass(text)
    response=getResponse(resultClass,intents)
    return response


def getResponse(resultClass, intents):
    listOfIntents=intents['intents']
    #print(listOfIntents)
    for i in listOfIntents:
        if i['tag']==resultClass:
            result=random.choice(i['responses'])
            break
    return result

#FOR TESTING ONLY
'''
if __name__ == '__main__':
    sentence="help please"
    result=chatbot_response(sentence)
    print(result)
'''
