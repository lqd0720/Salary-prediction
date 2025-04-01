import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Data Loading and Preprocessing
df = pd.read_csv('Salary Data.csv')

# Remove missing data
df = df.dropna()

# Experience Binning
exp_bins = [-1, 2, 5, 10, 15, 100]
exp_labels = ['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
df['Experience_Group'] = np.select(
    [
        (df['Years of Experience'] >= exp_bins[0]) & (df['Years of Experience'] < exp_bins[1]),
        (df['Years of Experience'] >= exp_bins[1]) & (df['Years of Experience'] < exp_bins[2]),
        (df['Years of Experience'] >= exp_bins[2]) & (df['Years of Experience'] < exp_bins[3]),
        (df['Years of Experience'] >= exp_bins[3]) & (df['Years of Experience'] < exp_bins[4]),
        (df['Years of Experience'] >= exp_bins[4])
    ],
    exp_labels,
    default='Unknown'
)

# Encode categorical features
categorical_features = ['Education Level', 'Job Title', 'Experience_Group']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features
FEATURES = [
    'Education Level',
    'Job Title',
    'Years of Experience',
    'Experience_Group'
]
TARGET = 'Salary'

X = df[FEATURES]
y = df[TARGET]

# Model Training and Evaluation
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror',  # Explicit regression objective
        eval_metric='rmse'
    ))
])

param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__max_depth': [3, 5]
}

pipeline.fit(X_train, Y_train)

# Predictions
Y_pred = pipeline.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")










