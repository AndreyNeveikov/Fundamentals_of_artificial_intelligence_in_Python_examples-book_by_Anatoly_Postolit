# Listing 6.24
Lr = LogisticRegression(C=1000.0, random_state=0)
Lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=Lr, test_idx=range(105, 150))
plt.xlabel('длина лепестка [стандартизированная]')
plt.ylabel('ширина лепестка [стандартизированная]')
plt.legend(loc='upper left')
plt.show()
