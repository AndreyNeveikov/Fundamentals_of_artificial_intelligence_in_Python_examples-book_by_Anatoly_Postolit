# Listing 8.15
# Modul Face_Lern2
import cv2
import os
import numpy as np
from PIL import Image

# Загрузка каскадов Хаара для поиска лиц
cascadePath = "C:\XML\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Формирование локального бинарного шаблона
# recognizer = cv2.createLBPHFaceRecognizer(1, 8, 8, 8, 123)
recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)


def get_images(path):
    # Ищем все фотографии и записываем их в image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.happy')]

    count = 0
    images = []
    labels = []

    for image_path in image_paths:
        # Переводим изображение в черно-белый формат и приводим его к формату массива
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
            cv2.imwrite('C:\\DataSet1\\user.' + str(subject_number) + '.' + str(count) +
                        '.jpg', image[y:y + h, x:x + w])
    return images, labels


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


