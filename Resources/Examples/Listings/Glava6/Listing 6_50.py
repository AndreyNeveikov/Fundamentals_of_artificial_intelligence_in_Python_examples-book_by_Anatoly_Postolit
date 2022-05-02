# Listing 6.50
predictions = model.predict(test_images)
ver1 = predictions[0]
im1 = np.argmax(predictions[0])
lab1 = test_labels[0]
print('Вероятность предсказаний для первого рисунка', ver1)
print('Первое изображение (после обучения)', im1)
print('Метка первого изображения', lab1)
