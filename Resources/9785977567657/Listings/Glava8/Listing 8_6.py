# Listing 8.6
# Modul Avto_Num_Cam2
import cv2
# загрузка предварительно обученной модели
faceCascade = cv2.CascadeClassifier('C:\XML\haarcascade_russian_plate_number.xml')
video_capture = cv2.VideoCapture(0)  # Активация видеокамеры
# Захват кадр за кадром
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    # Рисование прямоугольников вокруг объектов
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Показ обработанного кадра
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Когда обработка закончена, закрыть все окна
cv2.destroyAllWindows()