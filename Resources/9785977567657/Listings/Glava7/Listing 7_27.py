# Listing 7.27
# Модуль Trening2_Znaki
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Im_Yolo_Trening")
trainer.setTrainConfig(object_names_array=["stop", "zebra"],
                       batch_size=4,
                       num_experiments=2,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
