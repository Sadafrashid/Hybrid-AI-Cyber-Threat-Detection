def extract_features(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
