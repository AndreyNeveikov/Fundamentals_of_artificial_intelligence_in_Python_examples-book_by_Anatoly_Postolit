# Listing 6.31
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# Создание модели
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# Второй сверточный слой
model.add(Conv2D(32, kernel_size=3, activation='relu'))
# Создаем вектор для полносвязной сети.
model.add(Flatten())
# Создадим однослойный персептрон
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
