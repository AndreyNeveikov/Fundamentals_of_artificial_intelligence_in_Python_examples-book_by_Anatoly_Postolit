# Listing 8.11
# Modul Person_Video
import cv2
import imutils

# Инициализация детектора человека
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('Person.mp4')  # Загрузка видео файла

while cap.isOpened():
    # Чтение видео потока
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(600, image.shape[1]))

        # Обнаружение всех областей на изображении, в которых есть пешеходы
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

        # Рисование прямоугольников на изображении
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Video", image)  # показ обработанного видео
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Ожидание нажатия клавиши 'q'
            break
    else:
        break

cap.release()             # Освободить видео поток
cv2.destroyAllWindows()   # закрыть все окна
