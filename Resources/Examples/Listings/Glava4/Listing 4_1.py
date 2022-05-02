# Listing 4.1
# Искусственный нейрон (персептрон)
def perceptron(Sensor):
    b = 7  # Порог функции активации
    s = 0  # Начальное значение суммы
    for i in range(15):  # цикл суммирования сигналов от сенсоров
        s += int(Sensor[i]) * weights[i]

    if s >= b:
        return True  # Сумма превысила порог
    else:
        return False  # Сумма меньше порога
