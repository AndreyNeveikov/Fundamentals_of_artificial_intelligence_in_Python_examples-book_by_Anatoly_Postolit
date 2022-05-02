# Listing 6.33
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
