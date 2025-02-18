# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Carregando o dataset Iris
iris = load_iris()
X = iris.data    # Atributos (features)
y = iris.target  # Classes (target)

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Inicializando o classificador KNN com k=5 (pode ser ajustado conforme o problema)
knn = KNeighborsClassifier(n_neighbors=5)

# Treinando o modelo (o treinamento consiste basicamente em armazenar os dados de treino)
knn.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliando o desempenho do modelo
print("Acurácia:", knn.score(X_test, y_test))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Visualizando os resultados (opcional)
# Para visualização, usaremos apenas duas features (por exemplo, comprimento e largura da sépala)
plt.figure(figsize=(8, 6))
# Plotando os pontos do conjunto de teste, coloridos pela classe prevista
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', label='Previsões')
# Sobrepondo os pontos com a classe real
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='cool', marker='x', label='Real')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("KNN - Classificação do Dataset Iris")
plt.legend(loc='best')
plt.show()
