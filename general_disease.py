import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

# Initialize JS for SHAP plots
shap.initjs()

# ---------------------------------------------------------
#                 CUSTOM CSS & STYLING
# ---------------------------------------------------------
def local_css():
    st.markdown("""
        <style>
        /* Main Headers */
        h1 {
            color: #00FFFF; 
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            text-shadow: 0px 0px 10px rgba(0, 255, 255, 0.3);
        }
        h3 {
            color: #E0E0E0;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            padding: 20px !important;
            border-radius: 12px !important;
            text-align: center !important;
            box-shadow: 0px 4px 10px rgba(0,255,255,0.2);
            color: #000 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #000000 !important;
            font-size: 2.8rem !important;
            font-weight: 800 !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #004d39 !important;
            font-weight: 700 !important;
        }

        .stSelectbox label {
            color: #00FFFF !important;
            font-weight: bold;
        }

        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #008B8B 0%, #00FFFF 100%);
            color: #000000;
            font-weight: bold;
            border: none;
            padding: 0.8rem;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0px 0px 15px rgba(0, 255, 255, 0.6);
            color: #000000;
        }

        .result-box {
            background-color: #FFFFFF !important;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #00FFFF;
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
            color: #000 !important;
        }
        .result-box * {
            color: #000 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def general_disease_page():
    local_css()

    st.markdown("<h1 style='text-align:center;'>General Illness Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#AAAAAA;'>Select your symptoms below to generate an AI-powered diagnosis.</p>", unsafe_allow_html=True)

    # -------------------- LOAD FILES --------------------
    base_path = "D:/Research work/Mplementation/General Diseases/"
    
    def load_pickle(filename):
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            return pickle.load(open(path, "rb"))
        elif os.path.exists(filename):
            return pickle.load(open(filename, "rb"))
        else:
            return None

    model = load_pickle("best_model.pkl")
    label_encoder = load_pickle("label_encoder.pkl")
    symptoms_list = load_pickle("symptom_list.pkl")

    # FIX: Prevent NameError even if old leftover code exists
    selector = None
    scaler = None
    pca = None

    if not model or not symptoms_list:
        st.error(f"Critical files missing. Please check paths. Looking in: {base_path}")
        return

    final_feature_names = list(symptoms_list)

    symptoms_dropdown = ["None"] + symptoms_list

    # ------------------- SYMPTOM INPUTS ------------------------
    with st.expander("Symptom Checker", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            s1 = st.selectbox("Symptom 1", symptoms_dropdown)
            s2 = st.selectbox("Symptom 2", symptoms_dropdown)
            s3 = st.selectbox("Symptom 3", symptoms_dropdown)
        with col2:
            s4 = st.selectbox("Symptom 4", symptoms_dropdown)
            s5 = st.selectbox("Symptom 5", symptoms_dropdown)

    selected = [s1, s2, s3, s4, s5]

    # ------------------- INPUT VECTOR -------------------
    input_vec = np.zeros(len(symptoms_list))
    for s in selected:
        if s != "None":
            try:
                idx = symptoms_list.index(s)
                input_vec[idx] = 1
            except:
                pass

    # ------------------- PREPROCESS -------------------
    def preprocess(v):
        return v.reshape(1, -1)

    # ==========================================================
    #                   PREDICTION + XAI
    # ==========================================================
    if st.button("Analyze Health Condition"):

        if all(s == "None" for s in selected):
            st.error("⚠️ Please select at least one symptom.")
            return

        try:
            processed = preprocess(input_vec)

            # -----------------------------------------
            # 1. PREDICTION
            # -----------------------------------------
            try:
                proba = model.predict_proba(processed)[0]
            except:
                proba = np.array([1.0])

            pred_raw = model.predict(processed)[0]

            if label_encoder:
                pred_label = label_encoder.inverse_transform([int(pred_raw)])[0]
            else:
                pred_label = f"Class {pred_raw}"

            st.markdown(f"""
            <div class="result-box">
                <h2>🧬 Diagnosis Result</h2>
                <h3 style="color:#00FFFF;">{pred_label}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence Score", f"{np.max(proba)*100:.1f}%")

            st.write("---")
            st.subheader("Why did the AI make this prediction?")
            st.caption("Explainable AI (SHAP) shows which symptoms influenced the prediction.")

            # ==================================================
            #                   SHAP
            # ==================================================
            with st.spinner("Running Explainable AI (SHAP)..."):

                background = np.zeros_like(processed)

                pred_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
                
                explainer = shap.KernelExplainer(pred_fn, background)
                shap_vals = explainer.shap_values(processed, nsamples=100)

                target_idx = int(pred_raw)
                shap_class = shap_vals[target_idx] if isinstance(shap_vals, list) else shap_vals[0]
                shap_class = np.array(shap_class).reshape(-1)

                explanation = shap.Explanation(
                    values=shap_class,
                    base_values=0,
                    data=processed[0],
                    feature_names=symptoms_list
                )

                # -------- SHAP WATERFALL PLOT ----------
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### SHAP Waterfall Plot")
                    fig, ax = plt.subplots(figsize=(7,5))
                    shap.plots.waterfall(explanation, show=False, max_display=10)
                    st.pyplot(fig)
                    plt.clf()

                # -------- SHAP BAR CHART ----------
                with col2:
                    st.markdown("### Top Contributing Symptoms")
                    df = pd.DataFrame({
                        "Feature": explanation.feature_names,
                        "SHAP": explanation.values,
                        "Importance": np.abs(explanation.values)
                    }).sort_values("Importance", ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(7,5))
                    colors = ['#00FFFF' if v > 0 else '#FF4B4B' for v in df.SHAP]
                    ax.barh(df.Feature, df.SHAP, color=colors)
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    plt.clf()

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    st.set_page_config(page_title="Disease Predictor", layout="wide")
    general_disease_page()
