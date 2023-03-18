#importing different Libraries 

import sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer

#Loading the Data set

data = load_breast_cancer()

#Data- labeling

label_names = data['target_names']
labels = data['target']
feature_name = data["feature_names"]
feature = data['data']

#Exploring Data Loaded in the model

print(label_names)
print(feature_name)

#analysing Data 

print(labels)
print(len(labels))

print(feature)

#Splitting Data in to test and train 

from sklearn.model_selection import train_test_split

train , test,train_labels,test_labels = train_test_split(feature,labels,test_size = 0.33,random_state = 42)

#applying Machine Learning Model 

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
models = gnb.fit(train,train_labels)

print(predictions)

#Finding Accuracy of the model trained

from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels,predictions))



