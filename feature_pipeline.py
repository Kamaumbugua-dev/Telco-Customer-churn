from sklearn.model_selection import train_test_split

class FeaturePipeline:
    def __init__(self, df, target_col="Actual_Churn"):
        self.df = df
        self.target_col = target_col

    def split(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col].map({"Yes": 1, "No": 0})
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
