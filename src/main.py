import os
import numpy as np
import pandas as pd
from random import random
# from sklearn.metrics import accuracy_score

from trainer import Trainer
from preprocesor import bagOfWords

from ann_prueba import ANN
from pybrain_ann import *
# from scikit_ann import *

def counter(tweet, features):
  result = [0 for i in range(25)]
  for i in range(len(features)):
    if features[i] in tweet:
      result[i] = 1
  return result

def getEntrenamiento(features):
  path = os.path.abspath("./tweets.xlsx")
  # xl = pd.ExcelFile("/home/alejandro/Universidad/Semestre 7/Ingenieria del Conocimiento/Proyecto3/ANN/tweets.xlsx")
  xl = pd.ExcelFile(path)
  df = xl.parse("tweets")

  df = df.sample(frac=1)
  entrenamientoDF = df[:477]
  testDF = df[478:]
  # tweets = muestra["Texto"]
  # clases = muestra["Label"]

  # print(muestra)
  # print(tweets)
  # print(clase)

  entrenamiento = ([],[])
  test = ([],[])

  for _, row in entrenamientoDF.iterrows():
    line = row["Texto"].lower()
    clase = [1,0] if row["Label"] == "Seleccionado" else [0,1]
    entrenamiento[0].append(counter(line,features))
    entrenamiento[1].append(clase)

  for _, row in testDF.iterrows():
    line = row["Texto"].lower()
    clase = [1,0] if row["Label"] == "Seleccionado" else [0,1]
    test[0].append(counter(line,features))
    test[1].append(clase)

  return entrenamiento, test

def permormTest(y_pred, test):
  yTest = np.array(test[1])
  xTest = test[0]
  # y_pred = ann.forwardProp(xTest)
  for i in y_pred:
    if(i[0]>=0.5):
      i[0] = 1
      i[1] = 0
    else:
      i[0] = 0
      i[1] = 1
  #print(yTest)
  #print(y_pred)
  print(accuracy_score(y_pred, yTest))

def guardarPesos(W1,W2,W3):
  np.save("w1", W1)
  np.save("w2", W2)
  np.save("w3", W3)

def cargarPesos():
  W1 = np.load("w1.npy")
  W2 = np.load("w2.npy")
  W3 = np.load("w3.npy")
  return W1, W2, W3

def main():
  features = [a for a,b in bagOfWords()]
  entrenamiento, test = getEntrenamiento(features)
  ann = ANN()
  print("Empezando entrenamiento")
  trainer = Trainer(ann)
  trainer.train(np.array(entrenamiento[0]) ,np.array(entrenamiento[1]))

  # W1,W2,W3 = cargarPesos()
  # ann.W1 = W1
  # ann.W1 = W2
  # ann.W1 = W3
  print("Entrenamiento finalizado con los siguientes pesos")
  print(trainer.N.W1)
  print(trainer.N.W2)
  print(trainer.N.W3)
  # guardarPesos(trainer.N.W1, trainer.N.W2, trainer.N.W3)
  print("Test:")
  permormTest(ann.forwardProp(test[0]), test)

def main_pybrain():
  features = [a for a,b in bagOfWords()]
  entrenamiento, test = getEntrenamiento(features)
  ann = crear_red()
  train_pybrain(ann, entrenamiento)
  y_pred = [ann.activate(x) for x in test[0]]
  permormTest(y_pred, test)

def main_scikit():
  features = [a for a,b in bagOfWords()]
  entrenamiento, test = getEntrenamiento(features)
  ann = ann_scikit()
  train_scikit(ann, entrenamiento[0])
  y_pred = [ann.predict(x) for x in test[0]]
  permormTest(y_pred, test)

if __name__ == '__main__':
  main()