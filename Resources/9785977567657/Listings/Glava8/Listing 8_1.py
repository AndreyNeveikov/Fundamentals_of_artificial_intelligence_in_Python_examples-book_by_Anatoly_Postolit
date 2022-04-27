# Listing 8.1
# Modul Face1
import cv2

# загрузка фотографии
pixels = cv2.imread('Test_Face.jpg')
cv2.imshow('Input photo', pixels)
# загрузка предварительно обученной модели
classifier = cv2.CascadeClassifier('C:\XML\haarcascade_frontalface_default.xml')
# выполнение распознавания лиц
bboxes = classifier.detectMultiScale(pixels)
# формирование прямоугольника вокруг каждого обнаруженного лица
for box in bboxes:
    # формирование координат
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # рисование прямоугольников
    cv2.rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)

cv2.imshow('Window with face detection', pixels)       # показать обработанное изображение
cv2.imwrite('Test_Face_det.jpg', pixels)   # сохранить обработанное изображение

cv2.waitKey(0)           # держать с изображением открытым, пока не нажата любая клавиша
cv2.destroyAllWindows()  # закрыть все окна
