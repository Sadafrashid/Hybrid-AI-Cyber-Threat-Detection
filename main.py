from src.preprocessing import load_and_clean_data
from src.feature_engineering import extract_features
from src.hybrid_model import HybridIDS

def main():
    print("Starting Hybrid AI-Based Intrusion Detection System...")

    data = load_and_clean_data("dataset/CICIDS2017_sample.csv")
    X, y = extract_features(data)

    model = HybridIDS()
    model.train(X, y)

    accuracy, f1 = model.evaluate(X, y)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()
