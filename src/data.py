import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df.values

def prepare_features(points):
    scaler = StandardScaler()
    features = scaler.fit_transform(points)
    return features, scaler