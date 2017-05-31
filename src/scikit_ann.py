from sklearn.naive_bayes import GaussianNB

def ann_scikit():
  return GaussianNB()

def train_scikit(ann, entrenamiento):
  ann.fit(entrenamiento[0], entrenamiento[1])


