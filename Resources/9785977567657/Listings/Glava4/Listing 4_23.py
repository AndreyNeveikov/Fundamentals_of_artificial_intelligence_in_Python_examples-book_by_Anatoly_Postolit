# Listing 4.23
# адаптивный линейный нейрон


class AdaptiveLinearNeuron(object):

    def __init__(self, rate=0.01, niter=10):
        self.rate = rate  # rate - Темп обучения (между 0.0 и 1.0)
        self.niter = niter  # niter - Проходы по тренировочному набору данных

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.cost = []
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        # Вычисление чистого входного сигнала
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        # Вычисление линейной активации
        return self.net_input(X)

    def predict(self, X):
        # Возвращает метку класса после единичного шага (предсказание)
        return np.where(self.activation(X) >= 0.0, 1, -1)
