# Listing 6.32
import matplotlib as plt
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
# Построение графика точности предсказания
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()
# Построение графика потерь (ошибок)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()
