
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("dataset/Housing.csv")

# Convert text values into numbers
data = pd.get_dummies(data, drop_first=True, dtype=int)

# Separate input and output
X = data.drop("price", axis=1)
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

# Save comparison
results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2"]
)
results_df.to_csv("model_metrics.csv", index=False)

# Select best model
best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(), "Model"
]
best_model = models[best_model_name]

# Save model and feature columns
joblib.dump(best_model, "model.joblib")
joblib.dump(X.columns.tolist(), "feature_columns.joblib")

print("Training completed successfully!")
print("Best Model:", best_model_name)
print(results_df)
