import numpy as np

class ANN(object):
  """Red Neuronal"""
  def __init__(self, arg):
    self.entradas = 1
    self.salidas = 2
    self.tamCapaOculta = 3

    self.W1 = np.random(self.entradas, self.tamCapaOculta)
    self.W2 = np.random(self.tamCapaOculta, self.salidas)

  def tanh(self, x):
    return np.tanh(x)

  def devTanh(self, x):
    return 1-(self.tanh(x))**2

  def softMax(self, x):
    return np.exp(x[0])/np.sum(x[1])

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
    output = 0