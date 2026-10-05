# AI-Based Sugarcane Irrigation Advisory System

This repository currently contains the first working prototype: a Streamlit
interface for synthetic sugarcane field data.

## Run locally

```bash
python -m pip install -r requirements.txt
python src/generate_synthetic_data.py
streamlit run app.py
```

## Important limitation

The current dataset and water-stress target are synthetic. The app demonstrates
the project workflow and must not be presented as a field-validated irrigation
system. A real sugarcane dataset and domain-reviewed target are required before
making agricultural performance claims.

## Local Ollama chatbot

Install Ollama from [ollama.com](https://ollama.com), then run:

```bash
ollama pull llama3.2:3b
ollama serve
streamlit run app.py
```

The chatbot uses local retrieval from `knowledge/irrigation_guidance.md` and
calls Ollama at `http://localhost:11434`. It is restricted to irrigation topics.