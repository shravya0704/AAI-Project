from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
ROW_COUNT = 5000
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_sugarcane_fields.csv"


def generate_dataset(row_count: int = ROW_COUNT, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    growth_stages = np.array(["Establishment", "Grand Growth", "Maturity"])
    soil_types = np.array(["Sandy Loam", "Loam", "Clay Loam"])
    irrigation_methods = np.array(["Drip", "Sprinkler", "Flood"])

    growth_stage = rng.choice(growth_stages, row_count, p=[0.25, 0.45, 0.30])
    soil_type = rng.choice(soil_types, row_count, p=[0.30, 0.45, 0.25])
    irrigation_method = rng.choice(irrigation_methods, row_count, p=[0.45, 0.30, 0.25])
    crop_age = rng.integers(30, 361, row_count)
    soil_moisture = np.clip(rng.normal(31, 10, row_count), 8, 58)
    temperature = np.clip(rng.normal(31, 4.5, row_count), 18, 44)
    humidity = np.clip(rng.normal(62, 14, row_count), 25, 95)
    rainfall = np.clip(rng.gamma(shape=1.4, scale=5.5, size=row_count), 0, 45)
    evapotranspiration = np.clip(rng.normal(5.2, 1.2, row_count), 2, 9)
    days_since_irrigation = rng.integers(0, 16, row_count)

    stage_demand = pd.Series(growth_stage).map(
        {"Establishment": 0.7, "Grand Growth": 1.0, "Maturity": 0.8}
    ).to_numpy()
    soil_retention = pd.Series(soil_type).map(
        {"Sandy Loam": 1.15, "Loam": 0.85, "Clay Loam": 0.65}
    ).to_numpy()
    method_effect = pd.Series(irrigation_method).map(
        {"Drip": -0.25, "Sprinkler": 0.0, "Flood": 0.15}
    ).to_numpy()

    # This is a simulated label, created for prototyping rather than observed ground truth.
    stress_score = (
        0.055 * (35 - soil_moisture)
        + 0.16 * (temperature - 28)
        + 0.035 * (55 - humidity)
        + 0.19 * (evapotranspiration - 4)
        + 0.12 * days_since_irrigation
        - 0.07 * rainfall
        + 0.9 * stage_demand * soil_retention
        + method_effect
        + rng.normal(0, 0.7, row_count)
    )
    stress_level = np.select(
        [stress_score < 1.8, stress_score < 3.3],
        ["Low", "Moderate"],
        default="High",
    )

    return pd.DataFrame(
        {
            "soil_moisture_pct": np.round(soil_moisture, 2),
            "temperature_c": np.round(temperature, 2),
            "humidity_pct": np.round(humidity, 2),
            "rainfall_24h_mm": np.round(rainfall, 2),
            "evapotranspiration_mm_day": np.round(evapotranspiration, 2),
            "crop_age_days": crop_age,
            "days_since_irrigation": days_since_irrigation,
            "growth_stage": growth_stage,
            "soil_type": soil_type,
            "irrigation_method": irrigation_method,
            "stress_level": stress_level,
        }
    )


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset):,} rows to {OUTPUT_PATH}")
    print(dataset["stress_level"].value_counts().sort_index().to_string())