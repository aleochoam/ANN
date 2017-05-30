import os
import pandas as pd
from ann import ANN
from trainer import Trainer
import numpy as np
from preprocesor import bagOfWords
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

  muestra = df.sample(n=477)
  # tweets = muestra["Texto"]
  # clases = muestra["Label"]

  # print(muestra)
  # print(tweets)
  # print(clase)

  entrenamiento = ([],[])
  for _, row in muestra.iterrows():
    line = row["Texto"].lower()
    clase = [1,0] if row["Label"] == "Seleccionado" else [0,1]
    entrenamiento[0].append(counter(line,features))
    entrenamiento[1].append(clase)
    # line = limpiar(row["Texto"])

  # print(entrenamiento)
  return entrenamiento

def main():
  features = features = [a for a,b in bagOfWords()]
  entrenamiento = getEntrenamiento(features)
  ann = ANN()
  trainer = Trainer(ann)
  # print(entrenamiento[0])
  trainer.train(np.array(entrenamiento[0]) ,np.array(entrenamiento[1]))

if __name__ == '__main__':
  main()