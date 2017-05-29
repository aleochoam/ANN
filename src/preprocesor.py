import re
import sys
import unicodedata
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
    print(key,value)
    probs[key] = (value/totalWords, 10)

  for key, value in probs.items():
    print(key,value)
  return probs

def bagOfWords(file):
  xl = pd.ExcelFile("/home/alejandro/Universidad/Semestre 7/Ingenieria del Conocimiento/Proyecto3/ANN/tweets.xlsx")
  df = xl.parse("tweets")

  seleccionados = (df.loc[df['Label'] == "Seleccionado"])
  no_seleccionados = (df.loc[df['Label'] == "no seleccionado"])

  probs_selesccionados = countWords(seleccionados)
  probs_no_selesccionados = countWords(no_seleccionados)


def main():
  bagOfWords("seleccionados.txt")


if __name__ == '__main__':
  main()