import os
import re
import sys
import unicodedata
import operator
from math import log

import pandas as pd

def limpiar(line):
    line = line.lower()
    # line = elimina_tildes(line)
    # line = re.findall(r"[\w']+", line)
    line = line.replace(" de ", "")
    line = line.replace(" a ", "")
    line = line.replace(" y ", "")
    line = line.replace(" la ", "")
    line = line.replace(" su ", "")
    line = line.replace(" en ", "")
    line = line.replace(" con ", "")
    line = line.replace(" el ", "")
    line = line.replace(" del ", "")
    line = line.replace(" el ", "")
    line = line.replace(" al ", "")
    line = line.replace(" of ", "")
    line = line.replace(" da ", "")
    line = line.replace(" se ", "")
    line = line.replace(" mi ", "")
    line = line.replace(" para ", "")
    line = line.replace(" que ", "")
    line = line.replace(" in ", "")
    line = line.replace(" un ", "")
    line = line.replace(" to ", "")
    line = line.replace(" las ", "")
    line = line.replace(" di ", "")
    return line

def elimina_tildes(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))


def countWords(rows):
  words = {}
  for _, row in rows.iterrows():
    # line = limpiar(row["Texto"])
    line = limpiar(row["Texto"])
    for word in line.split():
      if word not in words:
        words[word] = 1
      else:
        words[word] += 1
  probs = {}
  totalWords = sum(words.values())
  for key, value in words.items():
    probs[key] = abs(value/totalWords)

  return probs

def bagOfWords():
  path = os.path.abspath("../tweets.xlsx")
  # xl = pd.ExcelFile("/home/alejandro/Universidad/Semestre 7/Ingenieria del Conocimiento/Proyecto3/ANN/tweets.xlsx")
  xl = pd.ExcelFile(path)
  df = xl.parse("tweets")

  seleccionados = (df.loc[df['Label'] == "Seleccionado"])
  no_seleccionados = (df.loc[df['Label'] == "no seleccionado"])

  probs_selesccionados = countWords(seleccionados)
  probs_no_selesccionados = countWords(no_seleccionados)
  diferencias = {}
  for key, value in probs_selesccionados.items():
      try:

        diferencias[key] = abs(probs_no_selesccionados[key] - value)
      except KeyError as e:
        diferencias[key] = abs(0 - value)

  for key, value in probs_no_selesccionados.items():
      if key not in probs_selesccionados.keys():
          diferencias[key] = abs(value - 0)

  sorted_diferencias = sorted(diferencias.items(), key=operator.itemgetter(1))
  return sorted_diferencias[-25:]


if __name__ == '__main__':
  pass