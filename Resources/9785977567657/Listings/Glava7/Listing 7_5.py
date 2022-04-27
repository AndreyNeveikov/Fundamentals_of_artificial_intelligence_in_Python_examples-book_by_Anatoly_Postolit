# Listing 7.5
# Модуль ImAi_5
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
im_path = os.path.abspath('C:\\ImAi\\')  # путь к папке с рисунками

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(im_path, "image5.jpg"),
                                             output_image_path=os.path.join(im_path, "image5_new.jpg"),
                                             minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
    print("--------------------------------")
