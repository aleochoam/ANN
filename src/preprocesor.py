import re
import sys
import unicodedata
import operator
from math import log

import pandas as pd

def limpiar(line):
    line = line.lower()
    line = elimina_tildes(line)
    line = re.findall(r"[\w']+", line)
    return line

def elimina_tildes(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))


def countWords(rows):
  words = {}
  for _, row in rows.iterrows():
    # line = limpiar(row["Texto"])
    line = row["Texto"].lower().split()
    for word in line:
      # print(word)
      if word not in words:
        words[word] = 2
      else:
        words[word] += 1

  probs = {}
  totalWords = sum(words.values())
  for key, value in words.items():
    probs[key] = abs(log(value/totalWords, 10))

  return probs

def bagOfWords(file):
  xl = pd.ExcelFile("/home/alejandro/Universidad/Semestre 7/Ingenieria del Conocimiento/Proyecto3/ANN/tweets.xlsx")
  df = xl.parse("tweets")

  seleccionados = (df.loc[df['Label'] == "Seleccionado"])
  no_seleccionados = (df.loc[df['Label'] == "no seleccionado"])

  probs_selesccionados = countWords(seleccionados)
  probs_no_selesccionados = countWords(no_seleccionados)

  diferencias = {}
  for key, value in probs_no_selesccionados.items():
    if key in probs_selesccionados.keys():
        diferencias[key] = value - probs_selesccionados[key]

  sorted_diferencias = sorted(diferencias.items(), key=operator.itemgetter(1))
  print(sorted_diferencias[:10])


def main():
  bagOfWords("seleccionados.txt")


if __name__ == '__main__':
  main()