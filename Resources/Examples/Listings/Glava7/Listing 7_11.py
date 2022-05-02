# Listing 7.11
# Модуль Im_Video4
from imageai.Detection import VideoObjectDetection
import os


def forFrame(frame_number, output_array, output_count):
    print("НОМЕР ФРЕЙМА ", frame_number)
    print("Массив параметров найденных объекта: ", output_array)
    print("Количество найденных объектов: ", output_count)
    print("------------КОНЕЦ ФРЕЙМА --------------")


execution_path = os.getcwd()
video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                      output_file_path=os.path.join(execution_path, "video_frame_analysis"),
                                      frames_per_second=20,
                                      per_frame_function=forFrame,
                                      minimum_percentage_probability=30)
