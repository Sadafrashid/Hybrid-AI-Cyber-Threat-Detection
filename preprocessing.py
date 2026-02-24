import pandas as pd

def load_and_clean_data(path):
    data = pd.read_csv(path)
    data = data.dropna()
    return data
