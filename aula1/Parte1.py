from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Carrega o dataset Iris
iris = load_iris()

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.7)

# Cria o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Treina o modelo
knn.fit(X_train, y_train)

# Faz previsões
y_pred = knn.predict(X_test)

# Avalia o modelo
accuracy = knn.score(X_test, y_test)
print(f"Acurácia: {accuracy}")