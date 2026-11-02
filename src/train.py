from sklearn.cluster import KMeans

def train_model(features, n_clusters, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(features)
    return model