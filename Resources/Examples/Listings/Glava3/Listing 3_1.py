# Listing 3.1
# Модуль Neuron
import numpy as np


# Создание класса нейрон
class Neuron:
    def __init__(self, w):
        self.w = w

    def y(self, x):  # Сумматор
        s = np.dot(self.w, x)  # Суммируем входы
        return s  # функция активации


Xi = np.array([2, 3])  # Задание значений входам
Wi = np.array([1, 1])  # Веса входных сенсоров
n = Neuron(Wi)  # Создание объекта из класса Neuron
print('S1= ', n.y(Xi))  # Обращение к нейрону
Xi = np.array([5, 6])  # Веса входных сенсоров
print('S2= ', n.y(Xi))  # Обращение к нейрону

Xi = np.array([1, 0, 0, 1])  # Задание значений входам
Wi = np.array([5, 4, 3, 1])  # Веса входных сенсоров
n = Neuron(Wi)  # Создание объекта из класса Neuron
print('S= ', n.y(Xi))  # Обращение к нейрону
