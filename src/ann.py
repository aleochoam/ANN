import numpy as np
from trainer import Trainer

class ANN(object):
  """Red Neuronal"""
  def __init__(self, arg):
    self.entradas = 2
    self.salidas = 2
    self.tamCapaOculta = 5
    print("Init")
    self.W1 = np.random.randn(self.entradas, self.tamCapaOculta)
    print(self.W1)
    self.W2 = np.random.randn(self.tamCapaOculta, self.salidas)
    print(self.W2)

  def tanh(self, x):
    return np.tanh(x)

  def devTanh(self, x):
    return 1-(self.tanh(x))**2

  def softMax(self, x):
    return self.softMax2(np.exp(x))

  def softMax2(self, x):
    sum = np.sum(x, axis=1)
    sum = np.transpose(np.tile(sum,(2,1)))
    return np.divide(x,sum)

  def devSoftMax(self, x):
    y = self.softMax(x)
    return y*(1-y)

  def ecm(self, i1, i2, o1, o2):
    return (1/2)*(i1-o1) + (1/2)*(i2-o2)

  def devEcm(self, i, o):
    return o-i

  def costFunction(self, X, y):
    #Compute cost for given X,y, use weights already stored in class.
    self.yHat = self.forwardProp(X)
    J = 0.5*sum((y-self.yHat)**2)
    return J

  def getParams(self):
      #Get W1 and W2 unrolled into vector:
      params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
      return params

  def setParams(self, params):
      #Set W1 and W2 using single paramater vector.
      W1_start = 0
      W1_end = self.tamCapaOculta * self.entradas
      self.W1 = np.reshape(params[W1_start:W1_end], (self.entradas , self.tamCapaOculta))
      W2_end = W1_end + self.tamCapaOculta*self.salidas
      self.W2 = np.reshape(params[W1_end:W2_end], (self.tamCapaOculta, self.salidas))

  def computeGradients(self, X, y):
    dJdW1, dJdW2 = self.costFunctionPrime(X, y)
    return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

  def costFunctionPrime(self, x, y):
    self.yHat = self.forwardProp(x)
    print("yHat:\n", self.yHat)
    delta3 = np.multiply(-(y-self.yHat), self.devSoftMax(self.z3))
    #print("delta3\n",delta3)
    dJdW2 = np.dot(self.a2.T, delta3)
    delta2 = np.dot(delta3, self.W2.T)* self.devTanh(self.z2)
    #print("delta2\n",delta2)
    #print("devTanh\n",self.devTanh(self.z2))
    dJdW1 = np.dot(x.T, delta2)
    return dJdW1, dJdW2

  #Forward Propagation
  def forwardProp(self, input):
    self.z2 = np.dot(input, self.W1)
    #print("z2\n",self.z2)
    self.a2 = self.tanh(self.z2)
    #print(self.a2)
    self.z3 = np.dot(self.a2, self.W2)
    #print(self.z3)
    outp = self.softMax(self.z3)
    return outp

def main():
  ann = ANN(1)
  #x = ann.forwardProp([[1],[2],[3]])
  x = np.array([[1,3],[2,3],[6,8],[4,5],[4,6],[6,7]])
  y = [[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]]
  trainer = Trainer(ann)
  trainer.train(x,y)
  #print("dJdW2:\n",ann.costFunction(x, y))
  # for i in range(10000):
  #   dJdW1, dJdW2 = ann.costFunctionPrime(x,y)
  #   #print("dJdW\n", dJdW1,"\n", dJdW2)
  #   alpha = 0.01
  #   ann.W1 = ann.W1 - alpha* dJdW1
  #   ann.W2 = ann.W2 - alpha* dJdW2
  #   #print("W1:\n",ann.W1)
  #   #print("W2:\n",ann.W2)
  # a = [[11,7],[4,8],[14,5],[2,4],[7,9],[20,21]]
  # b = np.array([[0,1],[0,1],[1,0],[0,1],[0,1],[1,0]])
  # classify = ann.forwardProp(a)
  # print(classify)
  #ann.forwardProp([1,2,3])
if __name__ == '__main__':
  main()