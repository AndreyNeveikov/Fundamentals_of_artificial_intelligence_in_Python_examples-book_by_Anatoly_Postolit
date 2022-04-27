# Listing 6.35
from keras.models import load_model
   # Запись обученной модели сети в файл 'my_model.h5
model.save('C:\mnist\my_model.h5')
   # Удаление модели.
del model
   # Загрузка обученной модели сети из файла
model = load_model('C:\mnist\my_model.h5')
