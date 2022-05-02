# Listing 8.12
import cv2
import os
import numpy as np
from PIL import Image

# Загрузка каскадов Хаара для поиска лиц
cascadePath = "C:\XML\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Формирование локального бинарного шаблона
recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
