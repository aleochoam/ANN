import os
import pandas as pd

def counter(tweet):
  features = ["moderno", "galerías", "#artemedellín", "botero", "memoria", \
            "museo", "casa", "galerías"]

  result = [0,0,0,0,0,0,0,0]
  for i in range(len(features)):
    if features[i] in tweet:
      result[i] = 1
  return result


def main():
  path = os.path.abspath("tweets.xlsx")
  # xl = pd.ExcelFile("/home/alejandro/Universidad/Semestre 7/Ingenieria del Conocimiento/Proyecto3/ANN/tweets.xlsx")
  xl = pd.ExcelFile(path)
  df = xl.parse("tweets")

  muestra = df.sample(n=477).head()
  tweets = muestra["Texto"]
  clase = muestra["Label"]

  print(muestra)
  print(tweets)
  print(clase)

  # print(list(seleccionados.head().sample(n=3)["Texto"]))

if __name__ == '__main__':
  main()