from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# Finds similar features that are highly correlated and remove it
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.keep_cols_ = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        # calculates the correlation matrix and takes absolutte values
        #  since negative values are also calculated
        corr = Xdf.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= self.threshold)]
        self.keep_cols_ = [c for c in Xdf.columns if c not in to_drop]
        return self

    # Applies transformation ad return result as pd dataframe
    def transform(self, X):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return Xdf[self.keep_cols_].copy()
