import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1500

data = {
    "antenna_array_size": np.random.choice([32, 64, 96, 128], n_samples),
    "transmission_power": np.random.uniform(10, 40, n_samples).round(2),
    "beamwidth": np.random.uniform(5, 60, n_samples).round(2),
    "channel_gain": np.random.uniform(-140, -60, n_samples).round(2),
    "interference_level": np.random.uniform(0, 1, n_samples).round(2),
    "user_density": np.random.randint(10, 1001, n_samples),
    "carrier_frequency": np.random.uniform(28, 83, n_samples).round(2),
    "mobility_speed": np.random.randint(0, 121, n_samples),
}

df = pd.DataFrame(data)
df["spectral_efficiency"] = (
    0.015 * df["antenna_array_size"]
    + 0.1 * df["transmission_power"]
    - 0.05 * df["beamwidth"]
    + 0.08 * (df["channel_gain"] + 140)
    - 5 * df["interference_level"]
    + 0.003 * df["user_density"]
    + 0.07 * df["carrier_frequency"]
    - 0.02 * df["mobility_speed"]
    + np.random.normal(0, 1.5, n_samples)
).round(2)

df["spectral_efficiency"] = df["spectral_efficiency"].clip(lower=1, upper=20)

df.to_csv("zmmwave_6g_continuous_dataset.csv", index=False)
print("✅ Dataset saved as zmmwave_6g_continuous_dataset.csv")
