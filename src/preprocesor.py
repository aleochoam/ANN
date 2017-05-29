import pandas as pd

def bagOfWords(file):
  xl = pd.ExcelFile("/home/alejandro/Universidad/Semestre 7/Ingenieria del Conocimiento/Proyecto3/ANN/tweets.xlsx")
  df = xl.parse("tweets")
  print(df.head())

def main():
  bagOfWords("seleccionados.txt")


if __name__ == '__main__':
  main()