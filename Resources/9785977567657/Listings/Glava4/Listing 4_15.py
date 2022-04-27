# Listing 4.15
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Описание класса Perceptron
class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # Темп обучения (от 0 до 1)
        self.n_iter = n_iter  # Количество итераций (уроков)

    '''
    Выполнить подгонку модели под тренировочные данные.
    Параметры
    X    - тренировочные данные: массив, размерность -  X[n_samples,
    n_features] , где 
                          n_samples - число образцов,
                          n_features - число признаков, 
    у - Целевые значения: массив, размерность - y[n_samples]
    Возвращает
    self: object
    '''
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # w_: 1-мерный массив – Веса после обучения
        self.errors_ = []  # errors_: список – ошибок классификации в каждой эпохе
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        return self

     # Рассчитать чистый вход
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Вернуть метку класса после единичного скачка
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
