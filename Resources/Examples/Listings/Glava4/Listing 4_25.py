# Listing 4.25
# Стандартизуем обучающую выборку
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# Обучение на стандартизованной выборке при rate = 0.01
aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(X_std,y)

# строим график зависимости стоимости ошибок от эпох обучения
plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')
plt.show()

# строим области принятия решений
plot_decision_regions(X_std, y, classifier=aln)
plt.title('ADALINE (градиентный спуск)')
plt.xlabel('длина чашелистика [стандартизованная]')
plt.ylabel('длина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show()
