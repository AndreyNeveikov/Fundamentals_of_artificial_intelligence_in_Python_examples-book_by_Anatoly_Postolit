# Listing 6.55
# Берем одну картинку из тестового набора данных.
img = test_images[0]
print(img.shape)

# Добавляем изображение в пакет данных, состоящий только из одного
элемента.
img = (np.expand_dims(img, 0))
print(img.shape)
predictions_single = model.predict(img)
print('Проверка на изображении из тестового набора данных')
print(predictions_single)
met = np.argmax(predictions_single[0])
print('Метка класса одежды', met)
