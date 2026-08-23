
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from openai import OpenAI
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------
# LOAD SAVED MODEL + PROJECT FILES
# ---------------------------------------------------------

MODEL_FILE = "readmission_xgboost_model.pkl"
THRESHOLD_FILE = "readmission_threshold.pkl"
RAG_CHUNKS_FILE = "rag_document_chunks.csv"

model = joblib.load(MODEL_FILE)
default_threshold = joblib.load(THRESHOLD_FILE)
chunks_df = pd.read_csv(RAG_CHUNKS_FILE)

openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it as a Hugging Face Space secret."
    )

client = OpenAI(api_key=openai_api_key)


# ---------------------------------------------------------
# DASHBOARD INPUT OPTIONS
# ---------------------------------------------------------

age_options = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

admission_type_options = [1, 2, 3, 4, 5, 6, 7, 8]

discharge_options = [
    1, 2, 3, 4, 5, 6, 7, 9, 10, 12,
    15, 16, 17, 18, 22, 23, 24, 25, 27, 28
]

a1c_options = [
    ">7",
    ">8",
    "Norm",
    "Not Recorded",
]

diabetes_med_options = [
    "No",
    "Yes",
]

diagnosis_options = [
    "Circulatory",
    "Diabetes",
    "Digestive",
    "Genitourinary",
    "Injury",
    "Musculoskeletal",
    "Neoplasms",
    "Other",
    "Respiratory",
    "Unknown",
]


# ---------------------------------------------------------
# READMISSION RISK FUNCTIONS
# ---------------------------------------------------------

def make_risk_plot(probability, threshold):
    fig, ax = plt.subplots(figsize=(7, 2.2))

    ax.barh(
        ["Patient"],
        [probability * 100],
        color="#4f86a6",
        height=0.45
    )

    ax.axvline(
        threshold * 100,
        color="#b44b4b",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.2f})"
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Estimated 30-Day Readmission Risk (%)")
    ax.set_title("Patient Risk Compared With Selected Threshold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()

    return fig


def create_risk_interpretation(probability, selected_threshold):
    prediction = int(
        probability >= selected_threshold
    )

    if prediction == 1:
        risk_label = "Additional Support Recommended"

        interpretation = (
            f"This patient's predicted readmission risk is "
            f"{probability * 100:.1f}%, which is above the "
            f"{selected_threshold:.2f} threshold. Additional "
            f"post-discharge support should be considered."
        )

    else:
        risk_label = "Standard Discharge Planning"

        interpretation = (
            f"This patient's predicted readmission risk is "
            f"{probability * 100:.1f}%, which is below the "
            f"{selected_threshold:.2f} threshold. Standard "
            f"discharge planning may be appropriate, while "
            f"clinical judgment should still be used."
        )

    return risk_label, interpretation


def predict_readmission(
    age,
    admission_type_id,
    discharge_disposition_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,
    A1Cresult,
    diabetesMed,
    diag_1_group,
    diag_2_group,
    diag_3_group,
    selected_threshold
):
    patient_df = pd.DataFrame({
        "age": [age],
        "admission_type_id": [admission_type_id],
        "discharge_disposition_id": [discharge_disposition_id],
        "time_in_hospital": [time_in_hospital],
        "num_lab_procedures": [num_lab_procedures],
        "num_procedures": [num_procedures],
        "num_medications": [num_medications],
        "number_outpatient": [number_outpatient],
        "number_emergency": [number_emergency],
        "number_inpatient": [number_inpatient],
        "number_diagnoses": [number_diagnoses],
        "A1Cresult": [A1Cresult],
        "diabetesMed": [diabetesMed],
        "diag_1_group": [diag_1_group],
        "diag_2_group": [diag_2_group],
        "diag_3_group": [diag_3_group],
    })

    probability = model.predict_proba(
        patient_df
    )[0, 1]

    risk_label, interpretation = create_risk_interpretation(
        probability,
        selected_threshold
    )

    risk_plot = make_risk_plot(
        probability,
        selected_threshold
    )

    return (
        f"{probability * 100:.1f}%",
        risk_label,
        interpretation,
        probability,
        risk_plot
    )


def update_threshold_result(
    raw_probability,
    selected_threshold
):
    if raw_probability is None:
        return (
            "Run a patient prediction first.",
            "Run a patient prediction first.",
            None
        )

    risk_label, interpretation = create_risk_interpretation(
        raw_probability,
        selected_threshold
    )

    risk_plot = make_risk_plot(
        raw_probability,
        selected_threshold
    )

    return (
        risk_label,
        interpretation,
        risk_plot
    )


# ---------------------------------------------------------
# RAG RETRIEVAL SYSTEM
# ---------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
RAG_MODEL = "gpt-5-nano"


def embed_texts(
    texts,
    model_name,
    batch_size=250
):
    texts = [
        str(text)
        for text in texts
    ]

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size
    ):
        batch = texts[
            start:start + batch_size
        ]

        response = client.embeddings.create(
            model=model_name,
            input=batch
        )

        all_embeddings.extend(
            item.embedding
            for item in response.data
        )

    return np.asarray(
        all_embeddings,
        dtype=np.float32
    )


chunk_embeddings = embed_texts(
    chunks_df["text"].tolist(),
    EMBEDDING_MODEL
)

document_index = NearestNeighbors(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

document_index.fit(
    chunk_embeddings
)


def retrieve_guidance(
    query,
    top_k=5
):
    query_embedding = embed_texts(
        [query],
        EMBEDDING_MODEL
    )

    distances, positions = (
        document_index.kneighbors(
            query_embedding,
            n_neighbors=top_k
        )
    )

    similarities = (
        1.0 - distances[0]
    )

    results = chunks_df.iloc[
        positions[0]
    ].copy()

    results["similarity"] = similarities

    results["retrieval_rank"] = range(
        1,
        len(results) + 1
    )

    return results


RAG_SYSTEM_PROMPT = """
You are a hospital readmission decision-support assistant.

Use only the retrieved AHRQ and CMS guidance provided in the prompt.

Provide concise, practical post-discharge recommendations for patients
who may be at risk of 30-day hospital readmission.

Requirements:
- Base every recommendation only on the retrieved evidence.
- Do not add recommendations that are not supported by the guidance.
- Include the source title and page number for every recommendation.
- Focus on discharge planning, follow-up, medication management,
  patient education, and care coordination.
- Do not diagnose the patient or replace clinical judgment.
- If the retrieved evidence does not support a recommendation,
  do not include it.
- Do not offer additional help or ask follow-up questions.
- End after the recommendations and source references.
""".strip()


def build_rag_prompt(
    query,
    retrieved_chunks
):
    evidence = []

    for _, row in retrieved_chunks.iterrows():
        evidence.append(
            f"Source: {row['title']} | "
            f"Page: {row['page']}\n"
            f"{row['text']}"
        )

    evidence_text = "\n\n".join(
        evidence
    )

    return f"""
Question:
{query}

Retrieved Guidance:
{evidence_text}

Using only the retrieved guidance, provide exactly 3 practical
post-discharge recommendations.

Prioritize the three recommendations that are most directly relevant
to the user's question. Avoid repetitive or overlapping recommendations.
Keep each recommendation concise.

For each recommendation:
1. State the recommended action.
2. Give one brief sentence explaining why it may help.
3. Cite the source title and page number.

Do not include information that is not supported by the retrieved evidence.
"""


def generate_rag_answer(query):
    if not query or not query.strip():
        return (
            "Enter a question about discharge planning, "
            "follow-up, medications, or readmission support."
        )

    retrieved = retrieve_guidance(
        query,
        top_k=5
    )

    prompt = build_rag_prompt(
        query,
        retrieved
    )

    response = client.responses.create(
        model=RAG_MODEL,
        instructions=RAG_SYSTEM_PROMPT,
        input=prompt,
        reasoning={
            "effort": "minimal"
        },
        max_output_tokens=700,
        store=False,
    )

    return response.output_text.strip()


# ---------------------------------------------------------
# COST / VALUE / RESOURCE IMPACT
# ---------------------------------------------------------

xgb_threshold_reference = pd.DataFrame({
    "Threshold": [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60
    ],
    "Patients Flagged": [
        19626,
        17555,
        12006,
        6891,
        2919
    ],
    "Readmissions Caught": [
        2253,
        2162,
        1779,
        1279,
        718
    ],
    "Readmissions Missed": [
        7,
        98,
        481,
        981,
        1542
    ]
})


def make_cost_plot(
    estimated_savings,
    intervention_cost,
    net_value
):
    labels = [
        "Estimated Savings",
        "Intervention Cost",
        "Net Value"
    ]

    values = [
        estimated_savings,
        intervention_cost,
        net_value
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        labels,
        values,
        color=[
            "#4f86a6",
            "#8ca6b5",
            "#2f6f8f"
        ]
    )

    ax.set_ylabel("Estimated Dollars")
    ax.set_title("Estimated Financial Impact")
    ax.ticklabel_format(
        style="plain",
        axis="y"
    )
    ax.grid(
        axis="y",
        alpha=0.15
    )

    plt.xticks(
        rotation=10
    )

    plt.tight_layout()

    return fig


def calculate_cost_impact(
    threshold,
    cost_per_readmission,
    cost_per_intervention,
    intervention_effectiveness_percent
):
    selected_row = xgb_threshold_reference[
        np.isclose(
            xgb_threshold_reference["Threshold"],
            float(threshold)
        )
    ]

    if selected_row.empty:
        raise ValueError(
            "Choose one of the tested thresholds: "
            "0.20, 0.30, 0.40, 0.50, or 0.60."
        )

    selected_row = selected_row.iloc[0]

    patients_flagged = int(
        selected_row["Patients Flagged"]
    )

    readmissions_caught = int(
        selected_row["Readmissions Caught"]
    )

    readmissions_missed = int(
        selected_row["Readmissions Missed"]
    )

    intervention_effectiveness = (
        intervention_effectiveness_percent
        / 100
    )

    expected_avoided = (
        readmissions_caught
        * intervention_effectiveness
    )

    estimated_savings = (
        expected_avoided
        * cost_per_readmission
    )

    intervention_cost = (
        patients_flagged
        * cost_per_intervention
    )

    net_value = (
        estimated_savings
        - intervention_cost
    )

    if intervention_cost > 0:
        roi = (
            net_value
            / intervention_cost
        ) * 100
    else:
        roi = 0

    if readmissions_caught > 0:
        break_even = (
            intervention_cost
            /
            (
                readmissions_caught
                * cost_per_readmission
            )
        ) * 100
    else:
        break_even = 0

    cost_plot = make_cost_plot(
        estimated_savings,
        intervention_cost,
        net_value
    )

    interpretation = (
        f"At a {float(threshold):.2f} threshold, the model flags "
        f"{patients_flagged:,} patients and identifies "
        f"{readmissions_caught:,} readmissions in the test data. "
        f"Under the selected assumptions, the estimated net value is "
        f"{net_value:,.0f} dollars. Lower thresholds generally identify "
        f"more readmissions but require more intervention resources."
    )

    return (
        f"{patients_flagged:,}",
        f"{readmissions_caught:,}",
        f"{readmissions_missed:,}",
        f"{expected_avoided:,.0f}",
        f"{intervention_cost:,.0f} dollars",
        f"{estimated_savings:,.0f} dollars",
        f"{net_value:,.0f} dollars",
        f"{roi:.1f}%",
        f"{break_even:.1f}%",
        interpretation,
        cost_plot
    )


# ---------------------------------------------------------
# DASHBOARD STYLE
# ---------------------------------------------------------

hospital_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate"
)

custom_css = """
.gradio-container {
    max-width: 1260px !important;
    margin: auto !important;
    background: #f7fafc;
}

.hospital-header {
    background: linear-gradient(135deg, #e7f2f8, #ffffff);
    border: 1px solid #d4e3ec;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

.threshold-box,
.info-card,
.guidance-card {
    background: #ffffff;
    border: 1px solid #d8e5ed;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
}

.threshold-box {
    background: #eef6fa;
    border-left: 4px solid #4f86a6;
}

button.primary {
    background: #2f6f8f !important;
    border-color: #2f6f8f !important;
}

button.primary:hover {
    background: #255c77 !important;
    border-color: #255c77 !important;
}

.tabs button.selected {
    color: #245f7d !important;
    border-color: #4f86a6 !important;
}

footer {
    display: none !important;
}
"""


# ---------------------------------------------------------
# BUILD DASHBOARD
# ---------------------------------------------------------

with gr.Blocks(
    theme=hospital_theme,
    css=custom_css,
    title="Hospital Readmission Decision Support"
) as demo:

    gr.HTML(
        """
        <div class="hospital-header">
            <div style="font-size: 36px; margin-bottom: 4px;">🏥</div>

            <h1 style="margin-bottom: 8px;">
                Hospital Readmission Decision Support
            </h1>

            <p style="font-size: 17px; margin-bottom: 5px;">
                Enter patient information to estimate 30-day readmission
                risk and help identify who may benefit from additional
                support after discharge.
            </p>

            <p style="color: #526777; margin-bottom: 0;">
                The dashboard combines XGBoost risk prediction,
                evidence-based discharge guidance, and resource-impact analysis.
            </p>
        </div>
        """
    )

    with gr.Tab("Readmission Risk"):

        with gr.Row():

            with gr.Column(scale=2):

                gr.Markdown(
                    """
                    ## Patient Information

                    Enter the patient's information below.
                    """
                )

                gr.Markdown("### Patient and Encounter Information")

                with gr.Group():

                    with gr.Row():
                        age = gr.Dropdown(
                            choices=age_options,
                            value="[70-80)",
                            label="Age"
                        )

                        admission_type_id = gr.Dropdown(
                            choices=admission_type_options,
                            value=1,
                            label="Admission Type"
                        )

                    discharge_disposition_id = gr.Dropdown(
                        choices=discharge_options,
                        value=1,
                        label="Discharge Disposition"
                    )

                gr.Markdown("### Hospital Stay")

                with gr.Group():

                    with gr.Row():
                        time_in_hospital = gr.Number(
                            value=5,
                            label="Days in Hospital"
                        )

                        num_lab_procedures = gr.Number(
                            value=40,
                            label="Lab Procedures"
                        )

                        num_procedures = gr.Number(
                            value=1,
                            label="Procedures"
                        )

                    with gr.Row():
                        num_medications = gr.Number(
                            value=15,
                            label="Medications"
                        )

                        number_diagnoses = gr.Number(
                            value=7,
                            label="Diagnoses"
                        )

                gr.Markdown("### Previous Healthcare Use")

                with gr.Group():

                    with gr.Row():
                        number_inpatient = gr.Number(
                            value=0,
                            label="Previous Inpatient Visits"
                        )

                        number_emergency = gr.Number(
                            value=0,
                            label="Previous Emergency Visits"
                        )

                        number_outpatient = gr.Number(
                            value=0,
                            label="Previous Outpatient Visits"
                        )

                gr.Markdown("### Diabetes and Diagnoses")

                with gr.Group():

                    with gr.Row():
                        a1c_result = gr.Dropdown(
                            choices=a1c_options,
                            value="Not Recorded",
                            label="A1C Result"
                        )

                        diabetes_med = gr.Dropdown(
                            choices=diabetes_med_options,
                            value="Yes",
                            label="Taking Diabetes Medication"
                        )

                    diag_1_group = gr.Dropdown(
                        choices=diagnosis_options,
                        value="Circulatory",
                        label="Primary Diagnosis"
                    )

                    with gr.Row():
                        diag_2_group = gr.Dropdown(
                            choices=diagnosis_options,
                            value="Diabetes",
                            label="Secondary Diagnosis"
                        )

                        diag_3_group = gr.Dropdown(
                            choices=diagnosis_options,
                            value="Other",
                            label="Additional Diagnosis"
                        )

            with gr.Column(scale=1):

                gr.Markdown("## Risk Settings")

                gr.HTML(
                    """
                    <div class="threshold-box">
                        <b>Recommended starting threshold: 0.40</b>
                        <br><br>
                        Lower thresholds identify more patients who may
                        need support, while higher thresholds flag fewer
                        patients and may be easier to manage when resources
                        are limited.
                    </div>
                    """
                )

                selected_threshold = gr.Slider(
                    minimum=0.20,
                    maximum=0.60,
                    value=default_threshold,
                    step=0.05,
                    label="Readmission Threshold"
                )

                predict_button = gr.Button(
                    "Check Readmission Risk",
                    variant="primary"
                )

                gr.Markdown("## Results")

                risk_probability = gr.Textbox(
                    label="Estimated 30-Day Readmission Risk"
                )

                risk_classification = gr.Textbox(
                    label="Support Recommendation"
                )

                risk_interpretation = gr.Textbox(
                    label="What This Means",
                    lines=5
                )

                risk_plot = gr.Plot(
                    label="Risk Visualization"
                )

                raw_probability = gr.Number(
                    visible=False
                )

        predict_button.click(
            fn=predict_readmission,
            inputs=[
                age,
                admission_type_id,
                discharge_disposition_id,
                time_in_hospital,
                num_lab_procedures,
                num_procedures,
                num_medications,
                number_outpatient,
                number_emergency,
                number_inpatient,
                number_diagnoses,
                a1c_result,
                diabetes_med,
                diag_1_group,
                diag_2_group,
                diag_3_group,
                selected_threshold
            ],
            outputs=[
                risk_probability,
                risk_classification,
                risk_interpretation,
                raw_probability,
                risk_plot
            ]
        )

        selected_threshold.change(
            fn=update_threshold_result,
            inputs=[
                raw_probability,
                selected_threshold
            ],
            outputs=[
                risk_classification,
                risk_interpretation,
                risk_plot
            ]
        )

    with gr.Tab("Discharge Support"):

        gr.Markdown(
            """
            ## Evidence-Based Discharge Support

            Ask a question about discharge planning, follow-up,
            medication management, care coordination, or reducing
            readmission risk.

            Recommendations are supported by guidance from
            **AHRQ and CMS**.
            """
        )

        rag_question = gr.Textbox(
            label="What would you like guidance on?",
            placeholder=(
                "Example: What follow-up support should be provided "
                "for a patient at high risk of readmission?"
            ),
            lines=3
        )

        rag_button = gr.Button(
            "Get Discharge Guidance",
            variant="primary"
        )

        rag_answer = gr.Markdown()

        rag_button.click(
            fn=generate_rag_answer,
            inputs=rag_question,
            outputs=rag_answer
        )

    with gr.Tab("Resource & Value Impact"):

        gr.Markdown(
            """
            ## Resource and Value Impact

            Adjust the scenario below to see how the readmission threshold
            and intervention assumptions affect hospital resources and
            estimated financial value.

            The threshold choices are limited to the values evaluated on
            the final XGBoost test set.
            """
        )

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("### Scenario Settings")

                cost_threshold = gr.Dropdown(
                    choices=[
                        0.20,
                        0.30,
                        0.40,
                        0.50,
                        0.60
                    ],
                    value=0.40,
                    label="Readmission Threshold"
                )

                cost_per_readmission_input = gr.Number(
                    value=16300,
                    label="Estimated Cost per Readmission"
                )

                cost_per_intervention_input = gr.Number(
                    value=200,
                    label="Cost per Patient Intervention"
                )

                intervention_effectiveness_input = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Expected Intervention Effectiveness (%)"
                )

                cost_button = gr.Button(
                    "Calculate Resource Impact",
                    variant="primary"
                )

                gr.Markdown(
                    """
                    The default assumptions match the project cost/value
                    analysis. These values can be changed to explore
                    different hospital scenarios.
                    """
                )

            with gr.Column(scale=2):

                gr.Markdown("### Estimated Impact")

                with gr.Row():
                    patients_flagged_output = gr.Textbox(
                        label="Patients Flagged"
                    )

                    readmissions_caught_output = gr.Textbox(
                        label="Readmissions Identified"
                    )

                    readmissions_missed_output = gr.Textbox(
                        label="Readmissions Missed"
                    )

                with gr.Row():
                    avoided_output = gr.Textbox(
                        label="Expected Readmissions Avoided"
                    )

                    intervention_cost_output = gr.Textbox(
                        label="Estimated Intervention Cost"
                    )

                with gr.Row():
                    savings_output = gr.Textbox(
                        label="Estimated Savings"
                    )

                    net_value_output = gr.Textbox(
                        label="Estimated Net Value"
                    )

                with gr.Row():
                    roi_output = gr.Textbox(
                        label="Estimated ROI"
                    )

                    break_even_output = gr.Textbox(
                        label="Break-Even Effectiveness"
                    )

                cost_interpretation_output = gr.Textbox(
                    label="What This Means",
                    lines=4
                )

                cost_plot_output = gr.Plot(
                    label="Financial Impact Visualization"
                )

        cost_button.click(
            fn=calculate_cost_impact,
            inputs=[
                cost_threshold,
                cost_per_readmission_input,
                cost_per_intervention_input,
                intervention_effectiveness_input
            ],
            outputs=[
                patients_flagged_output,
                readmissions_caught_output,
                readmissions_missed_output,
                avoided_output,
                intervention_cost_output,
                savings_output,
                net_value_output,
                roi_output,
                break_even_output,
                cost_interpretation_output,
                cost_plot_output
            ]
        )

    with gr.Tab("About & Limitations"):

        gr.Markdown(
            """
            ## About This Tool

            This dashboard was developed as a hospital readmission
            decision-support prototype using the UCI Diabetes
            130-US Hospitals dataset.

            The final prediction model is an XGBoost model using
            16 patient and hospital encounter predictors.

            ## Appropriate Use

            This tool is designed to help identify patients who may
            benefit from additional support after discharge.

            It should support clinical and operational decision-making,
            not replace clinical judgment.

            ## Limitations

            - The model was trained using historical hospital data.
            - The dataset focuses on patients with diabetes.
            - Model performance may differ with newer or different patient populations.
            - False positives and false negatives are still possible.
            - Cost and savings estimates are based on project assumptions.
            - The RAG knowledge base uses a limited set of AHRQ and CMS documents.
            - RAG recommendations depend on the retrieved source text and should be reviewed before real-world use.
            - Recommendations should be reviewed by appropriate healthcare professionals before use in practice.
            """
        )


if __name__ == "__main__":
    demo.launch()
