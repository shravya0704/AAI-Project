# Synthetic dataset

`synthetic_sugarcane_fields.csv` is generated locally by:

```bash
python src/generate_synthetic_data.py
```

The dataset is intentionally synthetic. It is a prototype for testing the ML and
Streamlit workflow, not evidence of real agricultural performance. The
`stress_level` target is created by a documented simulation using soil moisture,
temperature, humidity, rainfall, evapotranspiration, crop stage, soil type,
irrigation method, and time since irrigation.