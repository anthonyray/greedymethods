# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 08:43:30 2014
Examen n°1 : Intervalles de confiance et méthodes gloutonnes
@author: reinette

"""
import statsmodels.api as sm
import numpy as np
import scipy.stats as sp
import sklearn as sk
import sklearn.linear_model as LinearModel
import pandas as pd


"""
Exercice n°1 : Tests dans le modèle gaussien

1) y = X*theta + eps
2)
"""

# Récupération des données 
data = sm.datasets.get_rdataset('airquality').data

# Nettoyage du jeu de données
data = data.dropna()


# Regression lineaire
columns_name = data.columns.values
y = data[columns_name[0]] # y contient les valeurs de l'Ozone
X = data[columns_name[1:]]

# Centrer et réduire les données
y = (y) / np.sqrt(np.var(y))
X = (X) / np.sqrt(np.var(X))

X = sm.add_constant(X)

# Question 3

MCO = LinearModel.LinearRegression(fit_intercept=True)
MCO.fit(X,y)

# Question 4
alpha_max=1e4
eps=1e-12

n_alphas=50
alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),num=n_alphas)

clf = LinearModel.RidgeCV(alphas=alphas,fit_intercept=True,normalize=False,cv=sk.cross_validation.KFold(data.shape[0],7,shuffle=False))
clf.fit(X, y)

print clf.alpha_

# Question 5
result = y - np.dot(X,MCO.coef_) - MCO.intercept_
noise_estimation = (1.0 / (X.shape[0] - np.linalg.matrix_rank(X))) * (np.linalg.norm(result))**2

print noise_estimation

# Question 6

a = sp.t(X.shape[0] - np.linalg.matrix_rank(X) - 1).ppf(0.05)
b = sp.t(X.shape[0] - np.linalg.matrix_rank(X) - 1).ppf(0.95)

inverse = np.linalg.inv(np.dot(X.T,X))
theta_hat = clf.coef_
Z = np.zeros(X.shape[1],1)
sigma = noise_estimation

for i in range(Z.shape[0]):
    Z[i,0] = theta_hat[i] / (np.sqrt( inverse[i,i]  ))


# Question 7 
pred = np.array([1,197,10,70,3,1 ])

"""
Exercice n°2
"""

def stpforward(Y,X,M):
    theta = np.zeros((X.shape[0],1)) 
    r = Y 
    i = 0 
    S = list()
    columns = X.columns
    interval = X.columns
    X = X / np.sqrt(np.var(X)) # Normalisation des colonnes de X
    while i < M:
        alphas = list()        
        for col,i in enumerate(interval):
            pds = np.abs(np.vdot(X[interval[i]],r))
            alphas.append(pds)
        i_max = np.argmax(np.array(alphas))
        S.append(i_max)
        interval = np.delete(interval,i_max)
        # Déterminer Xs
        Xs = np.zeros(X.shape)
        
        theta_s = np.dot(np.linalg.pinv(Xs),Y)
        r = Y - np.dot(X,)
    
    return X
            
        
def stepforward(X,y,M):
    X = np.array(X)    
    theta = np.zeros((X.shape[1],1))        
    r = y
    i = 0
    S = list()
    interval = range(X.shape[1])
    X = X / np.sqrt(np.var(X))
    while i<M:
        alphas = list()
        for idx in interval:
            pds = np.abs(np.vdot(X[:,idx],r))
            alphas.append([idx,pds])
        alphas = np.array(alphas)
        i_max = alphas[np.argmax(alphas[:,1]),0]
        S.append(i_max)
        # Construction de Xs
        Xs = np.zeros(X.shape)
        for index in range(X.shape[1]):
            if index in S:
                Xs[:,index] = X[:,index]
        theta_S = np.dot(np.linalg.pinv(Xs),y)
        r = y - np.dot(X,theta_S)
        i = i + 1
    
    return theta_S,S
            
print stepforward(X,y,3)
print stepforward(X,y,4)
print stepforward(X,y,5)            
        