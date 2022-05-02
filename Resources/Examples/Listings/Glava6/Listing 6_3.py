# Listing 6.3
# Модуль PyBr_Dset2
import pybrain3
from pybrain3.datasets import SupervisedDataSet
ds = SupervisedDataSet(2, 1)

xorModel = [
   [(0, 0), (0,)],
   [(0, 1), (1,)],
   [(1, 0), (1,)],
   [(1, 1), (0,)],
]

for input, target in xorModel:
    ds.addSample(input, target)

print(ds)
