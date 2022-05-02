# Listing 6.16
# Модуль SciLearn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
# import species as species
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print("Тип массива data: {}".format(type(iris_dataset['data'])))
print("Форма массива data: {}".format(iris_dataset['data'].shape))
print("Цель: {}".format(iris_dataset['target']))
print("Названия ответов: {}".format(iris_dataset['target_names']))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
print("Расположение файла: \n{}".format(iris_dataset['filename']))
print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
print("Правильные ответы:\n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("Размерность массива X_train: {}".format(X_train.shape))
print("Размерность массива y_train: {}".format(y_train.shape))
print("Размерность массива X_test: {}".format(X_test.shape))
print("Размерность массива y_test: {}".format(y_test.shape))
'''
knn = KNeighborsClassifier(n_neighbors=1)
z = knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))
pr = knn.predict(X_new)
print("Метка сорта цветка: {}".format(pr))
print("Сорт цветка: {}".format(iris_dataset['target_names'][pr]))
'''

# создание и обучение классификатора
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# Практическое использование классификатора
X_new = np.array([[5, 2.9, 1, 0.2]])
pr = knn.predict(X_new)
# print("Метка сорта цветка: {}".format(pr))
print("Сорт цветка: {}".format(iris_dataset['target_names'][pr]))

pr = knn.predict(X_test)
print("Прогноз сорта на тестовом наборе:\n {}".format(pr))
print("Точность прогноза на тестовом наборе: {:.2f}".format(np.mean(pr == y_test)))



# Матрица расеивания с библиотекой seaborn
df = sb.load_dataset('iris')
sb.set_style("ticks")
sb.pairplot(df, hue='species', diag_kind="kde", kind="scatter", palette="husl")
plt.show()


'''
ds = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# sns.set(style="ticks", color_codes=True)
sns.pairplot(ds,  size=2)
plt.show()
'''

# создаем матрицу рассеяния из dataframe , цвет точек задаем с помощью y _ train
# pd.plotting.scatter_matrix(df, c=y_train, figsize=[8, 8], s=80, marker='D')
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10))

# grr = pd.splotting.scatter_matrix(df, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mgl.cm3)
# print(iris_dataset)

# df = pd.read_csv('C:\CSV\Iris.csv', header=None)
# print('Данные о цветках Ирис')
# print(df.to_string())

'''
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

# Создаем 2 D массив NumPy с единицами по главной диагонали и нулями в остальных ячейках
eye = np.eye(4)
print("массив NumPy:\n{}".format(eye))

# Преобразовываем массив NumPy в разреженную матрицу SciPy в формате CSR
sparse_matrix = sparse.csr_matrix(eye)
print("\nразреженная матрица SciPy в формате CSR:\n{}".format(sparse_matrix))

# разряженная матрица формата COO
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("формат COO:\n{}".format(eye_coo))

# Генерируем последовательность чисел от - 10 до 10 с 100 шагами
x = np.linspace(-10, 10, 100)
# Создаем второй массив с помощью синуса
y = np.sin(x)
# Функция создает линейный график на основе двух массивов
plt.plot(x, y, marker=".")
plt.show()

data = {'Имя': ["Дима", "Анна", "Петр", "Вика"],
        'Город': ["Москва", "Курск", "Псков", "Воронеж"],
        'Возраст': [24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
print(data_pandas)


x = np.array([[1, 2, 3], [4, 5, 6]])
print('Массив Х')
print(x)
'''
