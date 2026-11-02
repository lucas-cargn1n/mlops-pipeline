from sklearn.metrics import silhouette_score

def evaluate_model(model, features):
    inertia = model.inertia_
    sil = silhouette_score(features, model.labels_)
    metrics = {
        "inertia": round(inertia, 4),
        "silhouette_score": round(sil, 4),
        "n_clusters": model.n_clusters
    }
    return metrics