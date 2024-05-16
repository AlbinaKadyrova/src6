

# Простая реализация логистической регрессии и метода k ближайших соседей

Этот репозиторий содержит пример простой реализации логистической регрессии и метода k ближайших соседей на языке Python с использованием библиотеки NumPy и scikit-learn.

## Описание

- `LogisticRegression`: Класс, реализующий логистическую регрессию с помощью метода градиентного спуска.
- `KNeighborsClassifier`: Класс, реализующий метод k ближайших соседей для классификации.



## Пример

python
# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Предсказание меток классов на тестовом наборе
y_pred_log_reg = log_reg.predict(X_test)

# Оценка точности модели логистической регрессии
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='weighted')
print("Precision of Logistic Regression:", precision_log_reg)

# Создание и обучение модели метода k ближайших соседей
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Предсказание меток классов на тестовом наборе
y_pred_knn = knn.predict(X_test)

# Оценка точности модели метода k ближайших соседей
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
print("Precision of KNN:", precision_knn)


