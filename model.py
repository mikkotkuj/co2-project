import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Datan lataus
df = pd.read_csv("data/GT_data_fullv2.csv")

X = df[["co2_ppm", "ilmamaara_ls"]]
y = df["henkilot"]

# 2. Datan jako (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# --- MODEL 1: LINEAR REGRESSION ---
lr_model = LinearRegression().fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_val_pred = lr_model.predict(X_val)
lr_test_pred = lr_model.predict(X_test)

lr_mse_train = mean_squared_error(y_train, lr_train_pred)
lr_mse_val = mean_squared_error(y_val, lr_val_pred)
lr_mse_test = mean_squared_error(y_test, lr_test_pred)

# --- MODEL 2: RANDOM FOREST REGRESSION ---
# random_state takaa toistettavuuden, n_estimators=100 on vakiomäärä puita
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

rf_mse_train = mean_squared_error(y_train, rf_train_pred)
rf_mse_val = mean_squared_error(y_val, rf_val_pred)
rf_mse_test = mean_squared_error(y_test, rf_test_pred)

# --- TULOSTUKSET ---
print("=== LINEAR REGRESSION ===")
print(f"Coefficients: {lr_model.coef_}")
print(f"Intercept: {lr_model.intercept_:.2f}")
print(f"Train MSE: {lr_mse_train:.2f}")
print(f"Validation MSE: {lr_mse_val:.2f}")
print(f"Test MSE: {lr_mse_test:.2f}\n")

print("=== RANDOM FOREST REGRESSION ===")
print(f"Train MSE: {rf_mse_train:.2f}")
print(f"Validation MSE: {rf_mse_val:.2f}")
print(f"Test MSE: {rf_mse_test:.2f}\n")

# --- KUVAAJA: ACTUAL VS PREDICTED (Molemmat mallit testidatalla) ---
plt.figure(figsize=(8, 5))

# Pisteet kahdella eri värillä
plt.scatter(
    y_test,
    lr_test_pred,
    color="blue",
    label="Linear Regression",
    alpha=0.7,
    s=60,
)
plt.scatter(
    y_test,
    rf_test_pred,
    color="green",
    label="Random Forest",
    marker="s",
    alpha=0.7,
    s=60,
)

# Ihanteellinen suora (y = x)
min_val = min(y_test.min(), lr_test_pred.min(), rf_test_pred.min())
max_val = max(y_test.max(), lr_test_pred.max(), rf_test_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    label="Ideal fit",
)

plt.xlabel("Actual number of occupants")
plt.ylabel("Predicted number of occupants")
plt.title("Model Comparison: Linear Regression vs. Random Forest")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("data/model_comparison.png")
plt.show()