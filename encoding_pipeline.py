from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class EncodingPipeline:
    def __init__(self):
        self.encoder = None
        self.categorical_cols = None

    def fit(self, X):
        self.categorical_cols = X.select_dtypes(include="object").columns.tolist()
        self.encoder = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)],
            remainder="passthrough"
        )
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)
