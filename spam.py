import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\user\\Documents\\Python\\Practises\\spam")

messages = pd.read_csv("Spam SMS Collection",sep = '\t',
                       names = ['label','comment'])

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

lem = WordNetLemmatizer()
corpus = []

for i in range(len(messages['comment'])):
    review = re.sub('[^a-zA-Z]',' ',messages['comment'][i])
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    add = ' '.join(review)
    corpus.append(add)
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

tf = TfidfVectorizer(max_features =2500)
X = tf.fit_transform(corpus).toarray()

import pickle

TF = pickle.dump(tf,open('TF.pkl','wb'))
X = pd.DataFrame(X)

Y = pd.get_dummies(messages['label'],drop_first =True)

Fullraw = pd.concat([X,Y],axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(Fullraw,test_size =0.3, random_state =123)

Train_X = Train.drop(['spam'],axis =1)
Train_Y = Train['spam']
Test_X = Test.drop(['spam'],axis =1)
Test_Y = Test['spam']

M1 = MultinomialNB().fit(Train_X,Train_Y)

Test_pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_pred,Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

pickle.dump(M1,open('model.pkl','wb'))
