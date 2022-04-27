# Listing 8.5
# Modul Auto_Nemer
import cv2

# загрузка фотографии
pixels = cv2.imread('Test_Numer.jpg')
cv2.imshow('Input photo', pixels)
classifier = cv2.CascadeClassifier('C:\XML\haarcascade_russian_plate_number.xml')

# выполнение распознавания объектов
bboxes = classifier.detectMultiScale(pixels)
# формирование прямоугольника вокруг каждого обнаруженного объекта
for box in bboxes:
    # формирование координат
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # рисование прямоугольников
    cv2.rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 3)

cv2.imshow('Window with object detection', pixels)       # показать обработанное изображение
cv2.imwrite('Test_Numer_det.jpg', pixels)   # сохранить обработанное изображение

cv2.waitKey(0)           # держать с изображением открытым, пока не нажата любая клавиша
cv2.destroyAllWindows()  # закрыть все окна