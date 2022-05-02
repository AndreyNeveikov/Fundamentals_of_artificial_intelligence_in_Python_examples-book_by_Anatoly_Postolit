# Listing 4.24
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# Обучение при rate = 0.01
aln1 = AdaptiveLinearNeuron(0.01, 10).fit(X,y)
ax[0].plot(range(1, len(aln1.cost) + 1), np.log10(aln1.cost), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('Сумма квадратичных ошибок')
ax[0].set_title('ADALINE Темп обучения  0.01')

# Обучение при rate = 0.0001
aln2 = AdaptiveLinearNeuron(0.0001, 10).fit(X,y)
ax[1].plot(range(1, len(aln2.cost) + 1), aln2.cost, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Сумма квадратичных ошибок')
ax[1].set_title('ADALINE Темп обучения 0.0001')
plt.show()
