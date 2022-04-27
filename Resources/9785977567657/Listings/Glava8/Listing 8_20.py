# Listing 8.20
# Modul Seek_Face
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_Mark.yml')
cascadePath = 'C:\XML\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Тип шрифта
font = cv2.FONT_HERSHEY_SIMPLEX
# возбудить ID счетчика
# id1 = 0

# Список имен для id
names = ['None', 'Mark']

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10),)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # print(id)

        # Проверяем что лицо распознано
        if (confidence < 100):
            id_obj = names[1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id_obj = names[0]
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id_obj), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
