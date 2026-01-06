from sklearn.ensemble import RandomForestClassifier
import joblib

class ModelPipeline:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def save_model(self, path="artifacts/model.pkl"):
        joblib.dump(self.model, path)
