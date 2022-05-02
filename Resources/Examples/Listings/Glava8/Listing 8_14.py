# Listing 8.14
# Путь к фотографиям
path = 'C:\\YaleFace\\yalefaces\\'
# Получаем лица и соответствующие им номера
images, labels = get_images(path)
cv2.destroyAllWindows()

# Обучаем программу распознавать лица
recognizer.train(images, np.array(labels))
# Сохраняем результат тренировки
recognizer.write('Yale_face2.yml')
print('Обучение закончено')
