from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

class HybridIDS:
    def __init__(self):
        self.rf = RandomForestClassifier()
        self.iforest = IsolationForest(contamination=0.1)
        self.svm = SVC()

    def train(self, X, y):
        self.rf.fit(X, y)
        self.svm.fit(X, y)
        self.iforest.fit(X)

    def evaluate(self, X, y):
        rf_pred = self.rf.predict(X)
        svm_pred = self.svm.predict(X)

        # Hybrid voting
        final_pred = (rf_pred + svm_pred) / 2
        final_pred = [1 if p > 0.5 else 0 for p in final_pred]

        accuracy = accuracy_score(y, final_pred)
        f1 = f1_score(y, final_pred)

        return accuracy, f1
