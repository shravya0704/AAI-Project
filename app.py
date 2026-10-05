from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.rag import ask_ollama, load_chunks, ollama_is_available, retrieve


DATA_PATH = Path(__file__).parent / "data" / "synthetic_sugarcane_fields.csv"
TARGET = "stress_level"
NUMERIC_FEATURES = [
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "rainfall_24h_mm",
    "evapotranspiration_mm_day",
    "crop_age_days",
    "days_since_irrigation",
]
CATEGORICAL_FEATURES = ["growth_stage", "soil_type", "irrigation_method"]


st.set_page_config(page_title="Sugarcane Irrigation Advisor", page_icon="🌱", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(data: pd.DataFrame) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(data[NUMERIC_FEATURES + CATEGORICAL_FEATURES], data[TARGET])
    return model


def advisory(stress_level: str, probability: float, rainfall: float, soil_moisture: float) -> tuple[str, str]:
    if stress_level == "High" and rainfall < 10 and soil_moisture < 25:
        return "Irrigation recommended", "High predicted stress, low soil moisture, and limited recent rainfall."
    if stress_level == "High" or stress_level == "Moderate":
        return "Monitor and reassess", "The model indicates possible stress. Recheck field conditions before irrigating."
    return "Irrigation not currently recommended", "The simulated field conditions indicate low water stress."


data = load_data()
model = train_model(data)
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "latest_probability" not in st.session_state:
    st.session_state.latest_probability = None
if "latest_recommendation" not in st.session_state:
    st.session_state.latest_recommendation = None

st.title("Sugarcane Irrigation Advisor")
st.caption("Synthetic-data prototype: the model is for workflow demonstration, not field deployment.")

with st.sidebar:
    st.subheader("Field conditions")
    soil_moisture = st.number_input("Soil moisture (%)", 8.0, 58.0, 25.0, 0.1)
    temperature = st.number_input("Temperature (°C)", 18.0, 44.0, 32.0, 0.1)
    humidity = st.number_input("Humidity (%)", 25.0, 95.0, 55.0, 0.1)
    rainfall = st.number_input("Rainfall in last 24 hours (mm)", 0.0, 45.0, 3.0, 0.1)
    evapotranspiration = st.number_input("Evapotranspiration (mm/day)", 2.0, 9.0, 5.5, 0.1)
    crop_age = st.number_input("Crop age (days)", 30, 360, 145)
    days_since_irrigation = st.number_input("Days since irrigation", 0, 15, 6)
    growth_stage = st.selectbox("Growth stage", ["Establishment", "Grand Growth", "Maturity"])
    soil_type = st.selectbox("Soil type", ["Sandy Loam", "Loam", "Clay Loam"])
    irrigation_method = st.selectbox("Irrigation method", ["Drip", "Sprinkler", "Flood"])
    analyze = st.button("Analyze field", type="primary", use_container_width=True)

st.subheader("Current advisory")
prediction = None
probability = None
recommendation = None
if analyze:
    field = pd.DataFrame(
        [
            {
                "soil_moisture_pct": soil_moisture,
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "rainfall_24h_mm": rainfall,
                "evapotranspiration_mm_day": evapotranspiration,
                "crop_age_days": crop_age,
                "days_since_irrigation": days_since_irrigation,
                "growth_stage": growth_stage,
                "soil_type": soil_type,
                "irrigation_method": irrigation_method,
            }
        ]
    )
    prediction = model.predict(field)[0]
    probability = float(model.predict_proba(field).max())
    recommendation, explanation = advisory(prediction, probability, rainfall, soil_moisture)
    st.session_state.latest_prediction = prediction
    st.session_state.latest_probability = probability
    st.session_state.latest_recommendation = recommendation
    col1, col2, col3 = st.columns(3)
    col1.metric("Water stress", prediction)
    col2.metric("Model confidence", f"{probability:.0%}")
    col3.metric("Advisory", recommendation)
    st.info(explanation)
else:
    st.info("Enter field conditions in the sidebar and select Analyze field.")

st.subheader("Irrigation assistant")
st.caption("Local RAG chatbot powered by Ollama. It answers irrigation questions only.")
with st.expander("Questions this assistant can answer"):
    st.markdown(
        """
        - Why is irrigation recommended for this field?
        - How do soil moisture and recent rainfall affect irrigation need?
        - How should I interpret a moderate water-stress result?
        - How does crop growth stage affect irrigation monitoring?
        - What should I check before irrigating after rainfall?
        - How do drip, sprinkler, and flood methods differ for irrigation?

        The assistant does not answer questions about model training, confidence,
        disease, fertilizer, yield, or unrelated topics.
        """
    )
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about this field's irrigation recommendation")
if question:
    prediction = st.session_state.latest_prediction
    probability = st.session_state.latest_probability
    recommendation = st.session_state.latest_recommendation
    if prediction is None:
        st.warning("Analyze the field first so the assistant has a current prediction.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        field_summary = (
            f"Water stress: {prediction}; model confidence: {probability:.0%}; "
            f"advisory: {recommendation}; soil moisture: {soil_moisture}%; "
            f"rainfall in last 24 hours: {rainfall} mm."
        )
        chunks = retrieve(question, load_chunks())
        if not chunks:
            answer = (
                "I can answer irrigation questions about soil moisture, rainfall, "
                "evapotranspiration, crop stage, time since irrigation, soil type, "
                "and irrigation methods. Try asking why irrigation was recommended "
                "for this field."
            )
        elif not ollama_is_available():
            answer = "Ollama is not reachable at localhost:11434. Start Ollama and load the configured local model first."
        else:
            try:
                answer = ask_ollama(question, "\n\n".join(chunks), field_summary)
            except Exception as error:
                answer = f"The local Ollama request failed: {error}"
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

with st.expander("Prototype details"):
    st.write(f"Training rows: {len(data):,}")
    st.write("The stress label was simulated from water-balance-style relationships and is not measured ground truth.")
    st.write("Supported input ranges are the ranges used to generate this prototype dataset.")