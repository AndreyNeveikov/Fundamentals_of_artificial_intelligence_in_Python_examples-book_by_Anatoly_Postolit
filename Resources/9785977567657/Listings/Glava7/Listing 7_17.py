# Listing 7.17
# Модуль Trening1_Znaki
from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("Im_Trening")
model_trainer.trainModel(num_objects=2,
                         num_experiments=40,
                         enhance_data=True,
                         batch_size=4,
                         show_network_summary=True,
                         save_full_model=True)