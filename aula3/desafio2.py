# Importando as bibliotecas necessárias
from itertools import cycle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Carregando o dataset Wine
data = load_wine()
X = data.data
y = data.target

# 2. Dividindo os dados estratificadamente
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Encontrando o melhor K com GridSearchCV
param_grid = {"n_neighbors": range(1, 20)}
grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_["n_neighbors"]
print("Melhor valor de K:", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Acurácia:", accuracy)
print(
    "Relatório de Classificação:\n",
    classification_report(y_test, y_pred, target_names=data.target_names),
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names,
)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_bin = knn.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = cycle(["blue", "red", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"Curva ROC da classe {data.target_names[i]} (AUC = {roc_auc[i]:.2f})",
    )
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC Multiclasse")
plt.legend(loc="lower right")
plt.show()

from sklearn.inspection import permutation_importance

result = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42)
feature_importance = result.importances_mean

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance, align="center")
plt.yticks(range(len(data.feature_names)), data.feature_names)
plt.xlabel("Importância")
plt.title("Importância das Features")
plt.show()

joblib.dump(knn, "knn_model.pkl")
print("Modelo salvo como 'knn_model.pkl'")

