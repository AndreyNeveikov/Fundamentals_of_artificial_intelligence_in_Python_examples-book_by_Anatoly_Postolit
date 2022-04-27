# Listing 6.11
# Модуль Fishing_Test
import pickle

fileObject = open('MyNet_Fish.txt', 'rb')
net2 = pickle.load(fileObject)
fileObject.close()

# Хорошие погодные условия
y = net2.activate([2, 3, 80, 1])
print('Y1=', y)

# Средние погодные условия
y = net2.activate([10, 7, 40, 3])
print('Y2=', y)

# Плохие погодные условия
y = net2.activate([20, 11, 10, 5])
print('Y3=', y)
