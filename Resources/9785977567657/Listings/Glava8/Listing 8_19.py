# Listing 8.19
# Modul Face_Lern
import cv2
import numpy as np
import os

path = 'C:\\DataSet\\'  # задние имени папки с набором тренировочных фото
recognizer = cv2.face.LBPHFaceRecognizer_create()


# функция чтения изображений из папки из папки с тренировочными фото
def getImagesAndLabels(path):
    # Создаем список файлов в папке patch
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face = []  # Тут храним масив картинок
    ids = []  # Храним id лица
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        # Переводим изображение, тренер принимает изображения в оттенках серого
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face.append(img)  # записываем тренировочное фото в масив
        # Получаем id фото из его названия
        id = int(os.path.split(imagePath)[-1].split(".")[2])
        ids.append(id)  # записываем id тренировочного фото в масив
    return face, ids


# чтение тренировочного набора фотографий из папки path
faces, ids = getImagesAndLabels(path)
# Тренируем модель распознования
recognizer.train(faces, np.array(ids))
# Сохраняем результат тренировки
recognizer.write('face_Mark.yml')
