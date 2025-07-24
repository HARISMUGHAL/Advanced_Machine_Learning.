import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_csv("telco-churn-dataset.csv")
df = df.dropna()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

target = 'Churn'
features = df.columns.drop(['customerID', target])
numerical_cols = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df[features].select_dtypes(include=['object', 'bool']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Map target column to 0 and 1
df[target] = df[target].map({'Yes': 1, 'No': 0})

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# GridSearchCV to tune hyperparameters
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Accuracy:", best_model.score(X_test, y_test))
print("Best Parameters:", grid_search.best_params_)

joblib.dump(best_model, "telco_churn_pipeline.pkl")
print("Model saved as 'telco_churn_pipeline.pkl'")
