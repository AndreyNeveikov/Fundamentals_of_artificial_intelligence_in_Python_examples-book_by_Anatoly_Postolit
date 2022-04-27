# Listing 5.4
# Модуль Net1
import numpy as np


# функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Описание класса - Нейрон
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


# Описание класса - Нейронная сеть из 3-х слоев
class OurNeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])  # веса (одинаковы для всех нейронов)
        bias = 0  # смещение (одинаково для всех нейронов)

        # формируем сеть из 3-х нейронов
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)  # Фомируем выход Y1 из нейрона h1
        out_h2 = self.h2.feedforward(x)  # Фомируем выход Y2 из нейрона h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))  # Фомируем выход Y из нейрона О1
        return out_o1


network = OurNeuralNetwork()  # Создаем объект СЕТЬ из класса "Наша нейронная сеть"
x = np.array([2, 3])  # формируем входные параметры для сети Х1=2, Х2=3
print('Y= ', network.feedforward(x))  # Передаем входы в сеть и получает результат
