# Listing 8.9
# Modul Varianty
import cv2

# загрузка фотографии
pixels = cv2.imread('Test_Numer.jpg')
cv2.imshow('Input photo', pixels)
# загрузка предварительно обученной модели
# classifier = cv2.CascadeClassifier('C:\XML\haarcascade_fullbody.xml')
# classifier = cv2.CascadeClassifier('C:\XML\haarcascade_upperbody.xml')
# classifier = cv2.CascadeClassifier('C:\XML\haarcascade_lowerbody.xml')
# classifier = cv2.CascadeClassifier('C:\XML\haarcascade_righteye_2splits.xml')
# classifier = cv2.CascadeClassifier('C:\XML\haarcascade_lefteye_2splits.xml'
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