---
title: Hospital Readmission Decision Support
emoji: 🏥
colorFrom: blue
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
---

# Hospital Readmission Decision Support

This Gradio app estimates 30-day readmission risk using a saved XGBoost pipeline, provides evidence-based discharge support using RAG with AHRQ/CMS guidance, and includes interactive resource and value analysis.

## Required Space Secret

Add the following secret in the Hugging Face Space settings:

- `OPENAI_API_KEY`

## Required Repository Files

- `app.py`
- `requirements.txt`
- `readmission_xgboost_model.pkl`
- `readmission_threshold.pkl`
- `rag_document_chunks.csv`

This tool is a decision-support prototype and is not intended to replace clinical judgment.
