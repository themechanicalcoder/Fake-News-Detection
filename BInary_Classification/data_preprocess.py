# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:19:57 2019

@author: Gourav
"""

import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
from nltk.corpus import stopwords
from numpy import savez_compressed
from numpy import asarray

train="C:\\Users\\Gourav\\liar_dataset\\train.tsv"
test="C:\\Users\\Gourav\\liar_dataset\\test.tsv"
valid="C:\\Users\\Gourav\\liar_dataset\\valid.tsv"


train_news=_news=pd.read_csv(train, sep='\t', header=None).iloc[:,[1,2]]
test_news=_news=pd.read_csv(test, sep='\t', header=None).iloc[:,[1,2]]
valid_news=pd.read_csv(valid, sep='\t', header=None).iloc[:,[1,2]]


labels_map = {
        'true': 'true',
        'mostly-true': 'true',
        'half-true': 'true',
        'false': 'false',
        'barely-true': 'false',
        'pants-fire': 'false'
    }
train_news[1] = train_news[1].map(labels_map)
test_news[1] = test_news[1].map(labels_map)
valid_news[1] = valid_news[1].map(labels_map)



print(train_news.head())
print(test_news.head())
print(valid_news.head())


#no null values found
def null_check(dataframe):
    
    print(dataframe.isnull().sum())
    print(dataframe.info())
    
null_check(train_news)
null_check(test_news)    
null_check(valid_news)

stop_words = stopwords.words('english')
stemmer =PorterStemmer()

def stem_tokens(tokens):
    stemmed=[]
    for word in tokens:
        stemmed.append(stemmer.stem(word))
    return stemmed

import string 
translator = str.maketrans('', '', string.punctuation)
def preprocess(data):
    data= data.translate(translator)
    data=data.split()
    tokens=[w.lower() for w in data]
    stemmed=stem_tokens(tokens)
    stemmed=[w for w in stemmed if w not in stop_words]
    return stemmed


def transform(dataframe):
    x=[]
    y=[]
    for i in dataframe[2]:
        x.append(preprocess(i))
    for i in dataframe[1]:
        if(i=="true"):
            y.append(1)
        else:
            y.append(0)
            
    return asarray(x),asarray(y)

def save(array,filename):
    


trainx,trainy=transform(train_news)
testx,testy=transform(test_news)
validx,validy=transform(valid_news)
savez_compressed('preprocessed.npz', trainx,trainy,testx,testy,validx,validy)


    
