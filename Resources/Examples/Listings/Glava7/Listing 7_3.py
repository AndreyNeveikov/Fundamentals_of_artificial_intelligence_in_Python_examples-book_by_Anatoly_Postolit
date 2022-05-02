# Listing 7.3
# Модуль ImAi_3
from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()   # путь к папке с проектом
im_path = os.path.abspath('C:\\ImAi\\')  # путь к папке с рисунками

# конфигурирование модели нейронной сети
multiple_prediction = ImagePrediction()
multiple_prediction.setModelTypeAsResNet()
multiple_prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
multiple_prediction.loadModel()

# пустой массив с файлами рисунков
all_images_array = []

all_files = os.listdir(execution_path)  # формирование массива со всми файлами в папке проекта
for each_file in all_files:             # формирование массива с файлами только рисунков
    if(each_file.endswith(".jpg") or each_file.endswith(".png")):
        all_images_array.append(each_file)

print('Путь к изображениям', im_path)
print('Файлы в папке', all_files)
print('Масив изображений', all_images_array)

# Запуск модели на поиск объектов в фалах рисунков
results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

# Вывод результатов (найденые объекты в рисунках)
for each_result in results_array:
    predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
    for index in range(len(predictions)):
        print(predictions[index], " : ", percentage_probabilities[index])
    print("-----------------------")

