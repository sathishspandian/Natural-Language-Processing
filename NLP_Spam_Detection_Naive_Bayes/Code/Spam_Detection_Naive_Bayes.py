# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:26:27 2019

@author: Sathish Kumar
"""

#Importing Dataset
import numpy as np
import pandas as pd

messages = pd.read_csv(r'E:\Learning\Python\NLP\Data\SMSSpamCollection', sep = '\t',
                       names =["label","message"])

#Data Cleaning and preprocessing

import re
import nltk
#nltk.download()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

corpus =[]

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review =[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Document Matrix or Bag Of Words model creation
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000)

X = cv.fit_transform(corpus).toarray()

y= pd.get_dummies(messages["label"])
y = y.iloc[:,1].values


#Train and Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state =0)


#Traning model using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
spam_detection_model = MNB.fit(X_train,y_train)

#prediction
y_pred = spam_detection_model.predict(X_test)

#Confusion Matrix
CM = pd.crosstab(y_test,y_pred)

from sklearn.metrics import confusion_matrix
cmskl = confusion_matrix(y_test,y_pred)


# Accuracy 
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)*100








