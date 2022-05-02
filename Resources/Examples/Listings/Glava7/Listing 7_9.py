# Listing 7.9
# Модуль Im_Video2
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()
im_path = os.path.abspath('C:\\ImAi_Video\\')  # путь к папке с видео файлами

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(im_path, "traffic.mp4"),
                                             output_file_path=os.path.join(im_path, "traffic_detected"),
                                             frames_per_second=20, log_progress=True)
print(video_path)
