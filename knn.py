import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from LDA import LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

#returns accuracy of KNN classifier
def KNN_accuracy(X_train,y_train,X_test,y_test):
    """calculates the accuracy of the Knn classifier
    inputs:
    X_train:
    y_train:
    X_test:
    y_test:

    output: accuracy score (float)
    
    """
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    predictions = (neigh.predict(X_test))

    acc = np.sum(predictions== y_test)/len(y_test)
    return acc

# def cross_val(X,y,n,cv):
#     neigh = KNeighborsClassifier(n_neighbors=n)
#     neigh.fit(X, y)
#     scores = cross_val_score(neigh, X, y, cv=cv)
#     return scores

def cross_val(X_train, y_train, n, cv):
    """performs cross validation on the knn classifier 
    X_train:
    y_train:
    n: number of neighbours
    cv: numer of splits?
    
    output: mean score from each split(?)
    """
    classifier = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(classifier, X_train, y_train, cv=cv)
    return np.mean(scores)  # Return the mean score




