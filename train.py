import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# -----------------------------
# SET PATHS
# -----------------------------
project_folder = os.path.dirname(__file__)  # folder containing train.py
data_path = os.path.join(project_folder, "Housing.csv")
model_path = os.path.join(project_folder, "house_price_model.joblib")
heatmap_path = os.path.join(project_folder, "correlation_heatmap.png")

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Place Housing.csv in the project folder.")

df = pd.read_csv(data_path)
print("Dataset loaded successfully. Shape:", df.shape)

TARGET = "price"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------------
# CORRELATION HEATMAP (SAVE TO FILE)
# -----------------------------
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap Of House Price Predictors")
plt.tight_layout()
plt.savefig(heatmap_path)  # save heatmap instead of showing
plt.close()
print(f"Correlation heatmap saved at: {heatmap_path}")

# -----------------------------
# PREPROCESSING
# -----------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL TRAINING FUNCTION
# -----------------------------
def train_model(estimator, name, params=None):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    if params:
        grid = GridSearchCV(pipe, params, cv=5, scoring="neg_root_mean_squared_error")
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        print(f"\n{name} Best Params:", grid.best_params_)
    else:
        best = pipe.fit(X_train, y_train)

    preds = best.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5  # fix for newer sklearn
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name} Results:")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    return best, rmse

# -----------------------------
# TRAIN MODELS
# -----------------------------
lr_model, lr_rmse = train_model(LinearRegression(), "Linear Regression")
ridge_model, ridge_rmse = train_model(Ridge(), "Ridge", {"model__alpha": [0.1, 1, 10, 50, 100]})
lasso_model, lasso_rmse = train_model(Lasso(max_iter=5000), "Lasso", {"model__alpha": [0.001, 0.01, 0.1, 1, 10]})

# -----------------------------
# SELECT BEST MODEL
# -----------------------------
model_rmses = {
    "Linear": lr_rmse,
    "Ridge": ridge_rmse,
    "Lasso": lasso_rmse
}

best_name = min(model_rmses, key=model_rmses.get)
print("\nBest Model:", best_name)

if best_name == "Linear":
    best_model = lr_model
elif best_name == "Ridge":
    best_model = ridge_model
else:
    best_model = lasso_model

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(best_model, model_path)
print(f"\n✅ Model saved successfully at: {model_path}")
