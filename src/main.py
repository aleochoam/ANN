import os
import numpy as np
import pandas as pd
from random import random
from sklearn.metrics import accuracy_score
from ann_prueba import ANN
from trainer import Trainer
from preprocesor import bagOfWords

from pybrain_ann import *

def counter(tweet, features):
  result = [0 for i in range(25)]
  for i in range(len(features)):
    if features[i] in tweet:
      result[i] = 1
  return result

def getEntrenamiento(features):
  path = os.path.abspath("../tweets.xlsx")
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

def main():
  features = [a for a,b in bagOfWords()]
  entrenamiento, test = getEntrenamiento(features)
  ann = ANN()
  trainer = Trainer(ann)
  trainer.train(np.array(entrenamiento[0]) ,np.array(entrenamiento[1]))
  yTest = np.array(test[1])
  xTest = test[0]
  y_pred = ann.forwardProp(xTest)
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

def main_pybrain():
  features = [a for a,b in bagOfWords()]
  ann = crear_red()
  entrenamiento, test = getEntrenamiento(features)
  train(ann, entrenamiento)
  #evaluar y comprar resultados


if __name__ == '__main__':
  main_pybrain()