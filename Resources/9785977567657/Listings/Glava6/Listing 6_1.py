# Listing 6.1
# Модуль PyBr_Net1
import pybrain3
from pybrain3.tools.shortcuts import buildNetwork
net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])
print('Y=', y)
a = net['bias']
b = net['in']
c = net['hidden0']
d = net['out']
print(a)
print(b)
print(c)
print(d)
