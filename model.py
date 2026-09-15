import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

gt_df = pd.read_csv("data/GT_data.csv")

X = gt_df[["co2_ppm", "ilmamaara_ls"]]
y = gt_df["henkilot"]

model = LinearRegression()
model.fit(X, y)


# Luodaan ruudukko CO2- ja ilmamääräarvoista
co2_range = np.linspace(
    gt_df["co2_ppm"].min(),
    gt_df["co2_ppm"].max(),
    50
)

ilmamaara_range = np.linspace(
    gt_df["ilmamaara_ls"].min(),
    gt_df["ilmamaara_ls"].max(),
    50
)

CO2, ILMAMAARA = np.meshgrid(co2_range, ilmamaara_range)

# Muutetaan ruudukko mallin tarvitsemaksi DataFrameksi
X_plot = pd.DataFrame({
    "co2_ppm": CO2.ravel(),
    "ilmamaara_ls": ILMAMAARA.ravel()
})

# Ennusteet
HENKILOT = model.predict(X_plot).reshape(CO2.shape)


# 3D-kuvaaja
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Mallin ennustama pinta
ax.plot_surface(
    CO2,
    ILMAMAARA,
    HENKILOT,
    alpha=0.5
)

# Oikeat mittauspisteet
ax.scatter(
    gt_df["co2_ppm"],
    gt_df["ilmamaara_ls"],
    gt_df["henkilot"],
    s=40,
    label="GT-mittaukset"
)

ax.set_xlabel("CO₂ (ppm)")
ax.set_ylabel("Ilmamäärä (L/s)")
ax.set_zlabel("Henkilömäärä")

ax.set_title("Lineaarisen regressiomallin ennuste")

plt.tight_layout()

plt.savefig("3d plot.png")
plt.show()