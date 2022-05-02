# Listing 4.18
# Первые 50 элементов обучающей выборки (Строки 0-50, столбцы 0,1)
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')
# Следующие 50 элементов обучающей выборки (Строки 50-100, столбцы 0,1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формирование названий осей и вывод графика на экран
plt.xlabel('длина чашелистика')
plt.ylabel('длина лепестка')
plt.legend(loc='upper left')
plt.show()