# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Inicializando o classificador Árvore de Decisão
# Utilizando o critério 'gini' e definindo a profundidade máxima para evitar overfitting
tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Treinando o modelo
tree.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = tree.predict(X_test)

# Avaliando o desempenho do modelo
print("Acurácia:", tree.score(X_test, y_test))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Visualizando a Árvore de Decisão
plt.figure(figsize=(12, 8))
plot_tree(tree,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Árvore de Decisão - Dataset Iris")
plt.show()
