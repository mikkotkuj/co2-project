import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Ladataan Ground Truth -data
gt_df = pd.read_csv("data/GT_data.csv")

X = gt_df[["co2_ppm", "ilmamaara_ls"]]
y = gt_df["henkilot"]

# 2. Opetetaan malli
model = LinearRegression()
model.fit(X, y)

# 3. Luodaan tasainen CO2-akseli ennusteille
co2_range = np.linspace(gt_df["co2_ppm"].min(), gt_df["co2_ppm"].max(), 100)

# Määritetään ilmanvaihdon tehotasot ja niitä vastaavat ilmamäärät (L/s)
tehotasot = {
    "Teho I (~57 L/s, <80 dB)": 57,
    "Teho II (~71 L/s, 80–81 dB)": 71,
    "Teho III (~91–100 L/s, >82 dB)": 91,
}

# 4. Piirretään kuvaaja
plt.figure(figsize=(9, 6))

# Piirretään toteutuneet GT-mittauspisteet
plt.scatter(
    gt_df["co2_ppm"],
    y,
    color="blue",
    s=40,
    zorder=5,
    label="GT-datapisteet",
)

# Piirretään jokaiselle tehotasolle oma ennustesuoransa
varit = ["green", "orange", "red"]
for (nimi, q_arvo), vari in zip(tehotasot.items(), varit):
    X_plot = pd.DataFrame({"co2_ppm": co2_range, "ilmamaara_ls": q_arvo})
    y_pred = model.predict(X_plot)

    plt.plot(
        co2_range,
        y_pred,
        color=vari,
        linewidth=2,
        label=f"Mallin ennuste: {nimi}",
    )

plt.xlabel("CO2 (ppm)", fontsize=12)
plt.ylabel("Henkilömäärä", fontsize=12)
plt.title(
    "Linear Regression: CO2 vs Henkilömäärä eri IV-tehotasoilla", fontsize=14
)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("tehotasot_plot.png", dpi=300, bbox_inches='tight')
plt.show()