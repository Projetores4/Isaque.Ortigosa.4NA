# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parte 1: Avaliação com K-Folds no Dataset Iris

# Carregando o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Definindo o K-Folds (n_splits = 5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inicializando o classificador (Árvore de Decisão)
model = DecisionTreeClassifier(random_state=42)

# Avaliando o modelo utilizando cross_val_score
scores = cross_val_score(model, X, y, cv=kf)
print("Scores dos 5 folds:", scores)
print("Score médio:", scores.mean())

# Parte 2: Diagrama Ilustrativo do K-Folds

# Para fins ilustrativos, criaremos um conjunto de dados simplificado com 20 amostras
n_samples = 20
n_splits = 5
kf_demo = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Criaremos uma matriz para armazenar a atribuição:
# Cada linha representa um fold e cada coluna um índice de amostra.
# Valor 1: a amostra foi utilizada como teste; 0: utilizada para treinamento.
assignment = np.zeros((n_splits, n_samples))

# Itera sobre os folds e marca os índices de teste com 1
for fold, (train_index, test_index) in enumerate(kf_demo.split(np.arange(n_samples))):
    assignment[fold, test_index] = 1

# Plotando o diagrama com um heatmap
plt.figure(figsize=(10, 5))
plt.imshow(assignment, cmap='cool', aspect='auto')
plt.xlabel("Índice da Amostra")
plt.ylabel("Fold")
plt.title("Diagrama de Atribuição dos Folds (1 = Teste, 0 = Treinamento)")
plt.colorbar(label='Atribuição (0: Treinamento, 1: Teste)')
plt.xticks(np.arange(n_samples))
plt.yticks(np.arange(n_splits))
plt.show()
