import numpy as np

class ANN(object):
  """Red Neuronal"""
  def __init__(self, arg):
    self.entradas = 1
    self.salidas = 2
    self.tamCapaOculta = 3

    self.W1 = np.random.randn(self.entradas, self.tamCapaOculta)
    self.W2 = np.random.randn(self.tamCapaOculta, self.salidas)

  def tanh(self, x):
    return np.tanh(x)

  def devTanh(self, x):
    return 1-(self.tanh(x))**2

  def softMax(self, x):
    return np.exp(x[0])

  def softMax2(self, x):
    sum = np.sum(x)
    return np.divide(x,sum)

  def devSoftMax(self, x):
    y = self.softMax(x)
    return y*(1-y)

  def ecm(self, i1, i2, o1, o2):
    return (1/2)*(i1-o1) + (1/2)*(i2-o2)

  def devEcm(self, i, o):
    return o-i

  #Forward Propagation
  def forwardProp(self, input):
    self.z2 = np.dot(input, self.W1)
    self.a2 = self.tanh(self.z2)
    self.z3 = np.dot(self.a2, self.W2)
    outp = self.softMax2(self.softMax(self.z3))
    return outp

def main():
  ann = ANN(1)
  print(ann.forwardProp([[1],[2],[3]]))

if __name__ == '__main__':
  main()