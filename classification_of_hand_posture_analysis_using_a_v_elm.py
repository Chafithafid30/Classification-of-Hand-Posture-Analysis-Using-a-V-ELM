# -*- coding: utf-8 -*-
"""Classification of Hand Posture Analysis Using a V-ELM

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFgNfrhY0Y0k-81JhEDaOIacl07RsNE1

<h1>1. Import Dataset</h1>
"""

!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JVZLOn1BJ7oKwMTaAgk4aM8tEAZsqpWI' -O MoCapHandPostures.csv

import pandas as pd
import numpy as np
import statistics

data = pd.read_csv('MoCapHandPostures.csv')
data = data.iloc[1: , :]

data

"""<h1>2. Imputasi Missing Value</h1>"""

data = data.replace('?',np.nan)

data

import random
def imputasi(df_input):
  list_columns = df_input.columns[2:]
  class_columns = df_input.columns[0]
  df_input.dropna(thresh=11, axis=1)
  for column in list_columns:
    df_input[column] = pd.to_numeric(df_input[column])
    val = df_input.groupby(class_columns)[column]
    df_input[column] = df_input[column].fillna(value=val.transform('mean'))
  if df_input.isnull().any().any():
    for column in list_columns:
      val = df_input.groupby('User')[column]
      df_input[column] = df_input[column].fillna(value=val.transform('mean'))
  return df_input

data_impu = imputasi(data)

data_impu

data_impu = data_impu.drop(columns=['User'])

def minmax(df_input):
  list_fitur = df_input.columns[1:]
  for fitur in list_fitur:
    grouper = df_input.groupby('Class')[fitur]
    maxes = grouper.transform('max')
    mins = grouper.transform('min')
    df_input = df_input.assign(fitur=(df_input[fitur] - mins)/(maxes - mins))
  #df_input = df_input.groupby('Class').transform(lambda x: (x - x.min()) / x.max()- x.min())
  #for fitur in list_fitur:
    #max = df_input.groupby('Class')[fitur].max()
    #min = df_input.groupby('Class')[fitur].min()
    #df_input[fitur] = (df_input[fitur]-min)/(max-min)
  return df_input

"""<h1>3. Pemisahan Data Latih</h1>"""

from sklearn.model_selection import train_test_split

train70, test30 = train_test_split(data_impu, test_size=0.3)
train80, test20 = train_test_split(data_impu, test_size=0.2)

trNom70 = minmax(train70)
trNom80 = minmax(train80)
teNom30 = minmax(test30)
teNom20 = minmax(test20)

from scipy import stats as s
class voteELM:
  W = []
  beta = []
  mape = []

  def __fitELM(self,train,h,b1,b2):
    t = train['Class']
    t = np.stack(t.values)
    train = train.drop(columns=['Class'])
    d = len(train.columns)
    X = train.values
    W = np.random.uniform(b1,b2, (h,d))
    Hinit = X @ W.T
    H = 1/(1+np.exp(-1*Hinit))
    Hplus = np.linalg.inv(H.T @ H) @ H.T
    beta = Hplus @ t
    y = H @ beta
    return W, beta, t, y

  def fit(self,train,k=5,h=4,b1=-0.5,b2=0.5):
    for i in range(k):
      wf, bf, tf, yf = self.__fitELM(train,h,b1,b2)
      self.W.append(wf)
      self.beta.append(bf)
      MAPE = sum(abs(yf-tf)/yf)*1/len(tf)
      self.mape.append(MAPE)

  def test(self,test):
    t = test['Class']
    t = np.stack(t.values)
    y = []
    yt = []
    vote = []
    Xt = test.drop(columns=['Class'])
    Xt = Xt.values

    for idx,e in enumerate(self.W):
      Hinit = Xt @ e.T
      H = 1/(1+np.exp(-1*Hinit))
      yt = H @ self.beta[idx]
      yt = np.around(yt)
      y.append(yt)

    yt = list(zip(*y))

    pred = []
    for x in yt:
      mode = int(s.mode(x)[0])
      pred.append(mode)
    return pred

import time

model = voteELM()
model1 = voteELM()

st1 = time.time()
model.fit(trNom80,5,500)
et1 = time.time()

st2 = time.time()
model1.fit(trNom70,5,500)
et2 = time.time()

pred = model.test(teNom20)
pred1 = model1.test(teNom30)

from sklearn.metrics import accuracy_score

y_test = teNom20['Class'].tolist()
y_test1 = teNom30['Class'].tolist()

print("Akurasi V-ELM dengan rasio pelatihan 80:20 : ",accuracy_score(y_test,pred))
print("Akurasi V-ELM dengan rasio pelatihan 70:30 : ",accuracy_score(y_test1,pred1))
print("Waktu Pelatihan V-ELM dengan rasio pelatihan 80:20 : ",et1-st1)
print("Waktu Pelatihan V-ELM dengan rasio pelatihan 70:30 : ",et2-st2)
print("Nilai MAPE pada Pelatihan 80:20 : ", model.mape[:5])
print("Nilai MAPE pada Pelatihan 70:30 : ", model1.mape[5:])