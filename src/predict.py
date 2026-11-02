import geopy.distance

def predict(model, scaler, lat, lng):
    point_scaled = scaler.transform([[lat, lng]])
    cluster = int(model.predict(point_scaled)[0])
    centroid_scaled = model.cluster_centers_[cluster]
    centroid_original = scaler.inverse_transform([centroid_scaled])[0]
    
    distance_km = geopy.distance.geodesic(
        (lat, lng),
        (centroid_original[0], centroid_original[1])
    ).km
    
    prediction = {
        "cluster": cluster,
        "distance_km": round(distance_km, 4)
    }
    return prediction