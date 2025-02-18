
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Cria dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Cria o modelo KMeans
kmeans = KMeans(n_clusters=4)

# Treina o modelo
kmeans.fit(X)

# Obtém os центróides dos clusters
centroids = kmeans.cluster_centers_

# Imprime os центróides
print(f"Centróides: {centroids}")