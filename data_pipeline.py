import pandas as pd

class DataPipeline:
    def __init__(self, path, target_col="Actual_Churn"):
        self.path = path
        self.target_col = target_col

    def load(self):
        df = pd.read_csv(self.path)

        # Drop rows with missing target
        if self.target_col in df.columns:
            df = df.dropna(subset=[self.target_col])

        # Optional: fill missing values for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Optional: fill missing values for categorical columns
        cat_cols = df.select_dtypes(include="object").columns
        df[cat_cols] = df[cat_cols].fillna("Missing")

        return df
