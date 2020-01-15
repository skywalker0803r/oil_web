import pandas as pd
from sklearn.svm import SVR
from tqdm import tqdm_notebook as tqdm
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import warnings;warnings.simplefilter('ignore')
from sklearn.pipeline import Pipeline
import joblib
import pickle

class custom_model(object):
  def __init__(self):
    self.x_cols = ['T10','T50','T90','N+A']
    self.y_cols = ['C5NP','C5IP','C5N','C6NP','C6IP','C6N','C6A','C7NP','C7IP','C7N','C7A',
                   'C8NP','C8IP','C8N','C8A','C9NP','C9IP','C9N','C9A','C10NP','C10IP','C10N','C10A']
    self.model_23 = {}
    for y_name in y_cols:
      self.model_23[y_name] = Pipeline([('scaler',StandardScaler()),('reg',SVR(C=0.3))])
  
  def feature_engineering(self,X):
    X = pd.DataFrame(X,columns=self.x_cols)
    #X['f1'] = X['T90']-X['T50']
    #X['f2'] = X['T50']-X['T10']
    return X
  
  def fit(self,X,y):
    X = self.feature_engineering(X)
    for y_name in tqdm(self.y_cols):
      self.model_23[y_name].fit(X,y[y_name])
      y_pred = self.model_23[y_name].predict(X)
      # 序列預測把y_pred併入X 
      X.loc[:,y_name] = y_pred
    X = X[self.x_cols]
  
  def predict(self,X):
    results = pd.DataFrame(index=[*range(len(X))],columns=self.y_cols)
    X = self.feature_engineering(X)
    for y_name in self.y_cols:
      y_pred = self.model_23[y_name].predict(X)
      results.loc[:,y_name] = y_pred
      # 序列預測把y_pred併入X
      X.loc[:,y_name] = y_pred
    X = X[self.x_cols]
    return results.values