# Listing 6.36
# Загрузка обученной модели сети из файла
model_New = load_model('C:\mnist\my_model.h5')
y_train_pr = model_New.predict_classes(X_train[:3], verbose=0)
print('Первые 3 символа: ', y_train[:3])
print('Первые 3 предсказания: ', y_train_pr[:3])
