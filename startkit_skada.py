from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from skada import CORALAdapter, make_da_pipeline
import pandas as pd
import numpy as np

def get_estimator():
    # Define a column transformer for preprocessing
    preprocessor = make_column_transformer(
        ("passthrough", ["age"]),
        (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ["gender"]),
    )

    # Define the pipeline
    pipe = make_da_pipeline(
        preprocessor,
        CORALAdapter(),
        RandomForestRegressor(n_estimators=50, random_state=42)
    )
    return pipe

import os
import problem
os.environ["RAMP_TEST_MODE"] = "1"

X_v, y = problem.get_train_data(start=0, stop=100)
X_m, y_m = problem.get_train_data(start=-101, stop=-1)
X_df = pd.concat([X_v, X_m], axis=0)
y = np.concatenate([y, y_m])

pipe = get_estimator()
pipe.fit(X_df, y)

from sklearn.model_selection import cross_validate
#cv_results = cross_validate(pipe, X_df, y, cv=5, return_train_score=True, scoring='neg_mean_squared_error')

# Print cross-validation results
print("Cross-validation results:")
#print(f"Test scores (Negative MSE): {cv_results['test_score']}")
#print(f"Mean test score (Negative MSE): {np.mean(cv_results['test_score'])}")

