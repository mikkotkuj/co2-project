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
