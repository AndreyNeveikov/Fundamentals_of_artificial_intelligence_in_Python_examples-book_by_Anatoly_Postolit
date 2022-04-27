# Listing 7.22
# Модуль Detect_Video
from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="holo.mp4",
                                      output_file_path=os.path.join(execution_path, "holo-detected"),
                                      frames_per_second=30,
                                      minimum_percentage_probability=40,
                                      log_progress=True)
