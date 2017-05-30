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

  muestra = df.sample(n=477)
  tweets = muestra["Texto"]
  clases = muestra["Label"]

  # print(muestra)
  # print(tweets)
  # print(clase)

  entrenamiento = []
  for _, row in muestra.iterrows():
    line = row["Texto"].lower()
    clase = 1 if row["Label"] == "Seleccionado" else 0
    entrenamiento.append((counter(line), clase))
    # line = limpiar(row["Texto"])

  print(entrenamiento)



if __name__ == '__main__':
  main()