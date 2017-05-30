from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def crear_red():
  return buildNetwork(25, 5, 2, bias=True, outclass=SoftmaxLayer)


def train(ann, entrenamiento):
  ds = SupervisedDataSet(25, 2)
  for i in range(len(entrenamiento[0])):
    ds.addSample((entrenamiento[0][i]),(entrenamiento[1][i]))

  trainer = BackpropTrainer(ann, ds)
  for i in range(50):
    output = trainer.train()

  return ann



if __name__ == '__main__':
  pass