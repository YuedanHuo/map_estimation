from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tensorflow.keras import layers, models
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Define CNN model for time-series data
def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(128, 5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Define a custom estimator that integrates the CNN with other features
class CombinedEstimator(BaseEstimator):
    def __init__(self):
        self.cnn_model = None
        self.rf = None

    def fit(self, X, y):

        ppg = np.vstack(X[:, 0])  # First column contains PPG arrays
        ecg = np.vstack(X[:, 1])  # Second column contains ECG arrays
        time_series_data = np.stack([ppg, ecg], axis=-1)  # Shape: (n_samples, n_timesteps, n_channels)
    
        self.cnn_model = build_cnn_model(input_shape=(time_series_data.shape[1], time_series_data.shape[2]))
        self.cnn_model.compile(optimizer='adam', loss='mean_squared_error')
        self.cnn_model.fit(time_series_data, y, epochs=10, batch_size=32)

        # Extract CNN features
        cnn_features = self.cnn_model.predict(time_series_data)

        # Extract and scale other features (age, gender, etc.)
        other_features = X[:,2:]
        other_features = StandardScaler().fit_transform(other_features)
        
        # Combine CNN features with other features
        combined_features = np.concatenate([cnn_features, other_features], axis=1)

        # Train a Random Forest model on the combined features
        self.rf = RandomForestRegressor()
        self.rf.fit(combined_features, y)

        return self

    def predict(self, X):

        ppg = np.vstack(X[:, 0])
        ecg = np.vstack(X[:, 1])
        time_series_data = np.stack([ppg, ecg], axis=-1)

        # Extract CNN features
        cnn_features = self.cnn_model.predict(time_series_data)

        # Extract and scale other features
        other_features = X[:, 2:]
        other_features = StandardScaler().fit_transform(other_features)

        # Combine CNN features with other features
        combined_features = np.concatenate([cnn_features, other_features], axis=1)

        # Use Random Forest to make prediction
        return self.cnn_model.predict(time_series_data)
    
class IgnoreDomain(CombinedEstimator):
    def fit(self, X, y):
        # Ignore the samples with missing target
        X = X[y != -1]
        y = y[y != -1]
        return super().fit(X, y)

    
def get_estimator():
    return make_pipeline(
        make_column_transformer(
            ("passthrough", ['ecg', 'ppg', "age",'height', 'weight', 'bmi']),
            (OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1
            ), ["gender", 'domain']),
            remainder='drop'
        ),
        IgnoreDomain()
    )

import os
import problem
os.environ["RAMP_TEST_MODE"] = "1"

X_v, y = problem.get_train_data(start=0, stop=100)
X_m, y_m = problem.get_train_data(start=-101, stop=-1)
X_df = pd.concat([X_v, X_m], axis=0)
y = np.concatenate([y, y_m])

pipe = get_estimator()

from sklearn.model_selection import cross_validate
cv_results = cross_validate(pipe, X_df, y, cv=5, return_train_score=True, scoring='neg_mean_squared_error')

# Print cross-validation results
print("Cross-validation results:")
print(f"Test scores (Negative MSE): {cv_results['test_score']}")
print(f"Mean test score (Negative MSE): {np.mean(cv_results['test_score'])}")

