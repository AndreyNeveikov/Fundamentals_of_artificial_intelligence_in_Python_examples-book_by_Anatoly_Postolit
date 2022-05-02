# Listing 6.4
# Модуль PyBr_Trener
import pybrain3
import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader


net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
print(ds)

trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()
