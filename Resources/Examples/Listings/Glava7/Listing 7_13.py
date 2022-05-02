# Listing 7.13
# Модуль Im_Video6
from imageai.Detection import VideoObjectDetection
import os
execution_path = os.getcwd()


def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("Секунда : ", second_number)
    print("Массив выходных данных каждого кадра ", output_arrays)
    print("Массив подсчета уникальных объектов в каждом кадре : ", count_arrays)
    print("Среднее количество уникальных объектов в последнюю секунду: ", average_output_count)
    print("------------КОНЕЦ ДАННЫХ В ЭТОЙ СЕКУНДЕ --------------")


video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel()


video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                      output_file_path=os.path.join(execution_path, "video_second_analysis"),
                                      frames_per_second=20,
                                      per_second_function=forSeconds,
                                      minimum_percentage_probability=90)