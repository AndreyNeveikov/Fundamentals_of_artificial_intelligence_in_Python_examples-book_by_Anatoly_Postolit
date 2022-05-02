# Listing 7.1
# Модуль ImAi_1
from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()  # путь к папке с проектом
im_path = os.path.abspath('C:\\ImAi\\image1.jpg')  # путь к папке с рисунками

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(im_path, result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
