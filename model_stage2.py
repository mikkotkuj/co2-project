import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

gt_df = pd.read_csv("data/GT_data_fullv2.csv")

X = gt_df[["co2_ppm", "ilmamaara_ls"]]
y = gt_df["henkilot"]

# 70 % train, 30 % temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 15 % validation, 15 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
# Plot 1: CO2 vs. People
plt.figure(figsize=(8, 5))
plt.scatter(X_train["co2_ppm"], y_train, label="Training data")
plt.scatter(X_val["co2_ppm"], y_val, label="Validation data")
plt.scatter(X_test["co2_ppm"], y_test, label="Test data")

plt.xlabel("CO₂ (ppm)")
plt.ylabel("Number of people")
plt.title("CO₂ vs. Number of People")
plt.legend()
plt.grid(True)
plt.savefig("data/model.png")
plt.show()


# ---------------------------------------------------------------------------
# Stage 2 additions
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Physically motivated feature: N = Q * (C - C_out) / G
C_OUT = 400  # ppm, CO2 reading with 0 people in the room
X_phys = X.copy()  # keeps co2_ppm and ilmamaara_ls (multidimensional input)
X_phys["co2_x_flow"] = gt_df["ilmamaara_ls"] * (gt_df["co2_ppm"] - C_OUT)

# Same split as above (same rows) for the extended feature set
X_phys_train = X_phys.loc[X_train.index]
X_phys_val = X_phys.loc[X_val.index]
X_phys_test = X_phys.loc[X_test.index]

# M1: linear regression on (CO2, airflow)   (Stage 1 model)
# M2: linear regression on (CO2, airflow, airflow * (CO2 - C_out))
# M3: polynomial regression, degree 2, on (CO2, airflow)
m1 = LinearRegression()
m2 = LinearRegression()
m3 = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression())

m1.fit(X_train, y_train)
m2.fit(X_phys_train, y_train)
m3.fit(X_train, y_train)

# repeated 5-fold CV on train + validation (robustness check)
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)
X_dev = pd.concat([X_train, X_val])
X_phys_dev = pd.concat([X_phys_train, X_phys_val])
y_dev = pd.concat([y_train, y_val])


def cv_rmse(model, X_dev_, y_dev_):
    scores = cross_val_score(model, X_dev_, y_dev_, cv=cv,
                             scoring="neg_root_mean_squared_error")
    return -scores.mean()


results = pd.DataFrame(
    {
        "model": ["M1 linear (CO2, airflow)",
                  "M2 linear (CO2, airflow, airflow*(CO2-400))",
                  "M3 polynomial deg 2"],
        "train MSE": [mean_squared_error(y_train, m1.predict(X_train)),
                      mean_squared_error(y_train, m2.predict(X_phys_train)),
                      mean_squared_error(y_train, m3.predict(X_train))],
        "val MSE": [mean_squared_error(y_val, m1.predict(X_val)),
                    mean_squared_error(y_val, m2.predict(X_phys_val)),
                    mean_squared_error(y_val, m3.predict(X_val))],
        "CV RMSE": [cv_rmse(m1, X_dev, y_dev),
                    cv_rmse(m2, X_phys_dev, y_dev),
                    cv_rmse(m3, X_dev, y_dev)],
    }
)
results["val RMSE"] = results["val MSE"] ** 0.5
print(results.round(3).to_string(index=False))

# Final model: M2 -> test error (test set used only here)
test_pred = m2.predict(X_phys_test)
test_mse = mean_squared_error(y_test, test_pred)
print("Test MSE:", round(test_mse, 3), "RMSE:", round(test_mse ** 0.5, 3))
print("Test within +-1 person after rounding:",
      np.mean(np.abs(np.round(test_pred) - y_test) <= 1))
print("M2: coefficients", dict(zip(X_phys.columns, m2.coef_)),
      "intercept", m2.intercept_)
print("CO2 per person implied by the co2_x_flow coefficient (L/h):",
      3600 / (m2.coef_[2] * 1e6))

# Plot 2: CO2 vs. People for the three fan levels
plt.figure(figsize=(5.2, 3.3))
for level, group in gt_df.groupby("ilmanvaihdon_teho"):
    plt.scatter(group["co2_ppm"], group["henkilot"], s=14,
                label=f"Fan level {level}")
plt.xlabel("CO₂ (ppm)")
plt.ylabel("Number of people")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/co2_vs_people.png", dpi=200)

# Plot 3: predicted vs. true for the three models
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, title, model, (Xa, Xb, Xc) in zip(
    axes,
    ["M1", "M2", "M3"],
    [m1, m2, m3],
    [(X_train, X_val, X_test),
     (X_phys_train, X_phys_val, X_phys_test),
     (X_train, X_val, X_test)],
):
    ax.scatter(y_train, model.predict(Xa), s=12, label="Train")
    ax.scatter(y_val, model.predict(Xb), s=12, label="Validation")
    ax.scatter(y_test, model.predict(Xc), s=12, label="Test")
    ax.plot([0, 42], [0, 42], "k--", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("True number of people")
    ax.grid(True)
axes[0].set_ylabel("Predicted")
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("data/pred_vs_true.png", dpi=200)