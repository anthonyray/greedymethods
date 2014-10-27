# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 08:43:30 2014
Examen n°1 : Intervalles de confiance et méthodes gloutonnes
@author: reinette

"""
import statsmodels.api as sm
import numpy as np
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
y = (y )/ np.sqrt(np.var(y))
X = (X ) /np.sqrt(np.var(X))


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
result = y - np.dot(X,MCO.coef_)
noise_estimation = (1.0 / (X.shape[0] - np.linalg.matrix_rank(X))) * (np.linalg.norm(result))**2

print noise_estimation

# Question 6


