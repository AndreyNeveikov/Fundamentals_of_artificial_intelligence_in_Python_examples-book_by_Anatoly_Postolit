# Listing 7.2
# Модуль ImAi_2
from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()  # путь к папке с проектом
im_path = os.path.abspath('C:\\ImAi\\image4.jpg')  # путь к папке с рисунками

prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(im_path, result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
