import math
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth radius in km
    return km

class SimpleEncoder:
    """Minimal fit/transform encoder for categorical cols returning pandas DataFrame columns."""
    def __init__(self):
        self.encoders = {}

    def fit(self, df, cols):
        for c in cols:
            self.encoders[c] = {v: i for i, v in enumerate(df[c].fillna('___NA___').unique())}

    def transform(self, df, cols):
        out = pd.DataFrame(index=df.index)
        for c in cols:
            mapping = self.encoders.get(c, {})
            out[c + '_enc'] = df[c].fillna('___NA___').map(mapping).fillna(-1).astype(int)
        return out

    def fit_transform(self, df, cols):
        self.fit(df, cols)
        return self.transform(df, cols)
