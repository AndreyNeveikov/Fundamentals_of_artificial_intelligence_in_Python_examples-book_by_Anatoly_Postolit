# Listing 8.13
def get_images(path):
    # Ищем все фотографии и записываем их в image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.happy')]

    count = 0
    images = []
    labels = []

    for image_path in image_paths:
        # Переводим изображение в черно-белый формат и приводим его к   формату массива
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        # Из каждого имени файла извлекаем номер человека, изображенного на фото
        subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        # Определяем области, где есть лица
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Если лицо нашлось, добавляем его в список images, а соответствующий ему номер в список labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)
            # В окне показываем изображение
            cv2.imshow("", image[y: y + h, x: x + w])
            cv2.waitKey(50)
            count += 1
            # Сохраняем лицо
            cv2.imwrite('C:\\DataSet1\\user.' + str(subject_number) + '.' + str(count) + '.jpg', image[y:y + h, x:x + w])
    return images, labels
