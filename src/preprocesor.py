def bagOfWords(file):
  file = open(file,"r")
  text = file.read().decode('utf8')
  for line in text:
    for word in line.split():
      for l in word:
        print(l)
        input()

def main():
  bagOfWords("seleccionados.txt")


if __name__ == '__main__':
  main()