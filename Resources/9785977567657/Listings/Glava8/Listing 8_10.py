# Listing 8.10
# Modul Person_Im
import cv2
import imutils

# Инициализация детектора человека
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Чтение изображения
image = cv2.imread('image_str.jpg')
cv2.imshow('Input photo', image)
# Изменение размера изоб ражения
image = imutils.resize(image, width=min(800, image.shape[1]))

# Обнаружение всех областей на изображении, где есть пешеходы
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

# Рисование прямоугольников на изображении
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Image", image)  # Отображение обработанного изображения
cv2.imwrite('image_str_det.jpg', image)   # сохранение обработаннго изображения

cv2.waitKey(0)           # Ожидание нажатия любой клавиши
cv2.destroyAllWindows()  # закрыть все окна
