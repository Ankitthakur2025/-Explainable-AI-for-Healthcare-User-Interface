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

        /* FIX: Metric card visibility */
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

        /* Selectbox Styling */
        .stSelectbox label {
            color: #00FFFF !important;
            font-weight: bold;
        }

        /* Analyze Button */
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

        /* Results Container */
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

    # Apply CSS
    local_css()

    st.markdown(
        "<h1 style='text-align:center;'>General Illness (common health conditions) Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align:center; color:#AAAAAA;'>Select your symptoms below to generate an AI-powered diagnosis.</p>", unsafe_allow_html=True)

    # -------------------- LOAD MODELS --------------------
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

    scaler = load_pickle("scaler.pkl")
    selector = load_pickle("selector.pkl")
    pca = load_pickle("pca.pkl")

    if not model or not symptoms_list:
        st.error(f"Critical files missing. Please check paths. Looking in: {base_path}")
        return

    # ----------------------------------------------------
    #  FIX: ENSURE selector ALWAYS EXISTS (NO NameError)
    # ----------------------------------------------------
    if selector is None:
        class DummySelector:
            def get_support(self): 
                return np.ones(len(symptoms_list), dtype=bool)
            def transform(self, X):
                return X
        selector = DummySelector()

    # ----------------------------------------------------
    #   FINAL FEATURE LIST AFTER FEATURE SELECTION
    # ----------------------------------------------------
    final_feature_names = list(symptoms_list)

    if hasattr(selector, "get_support"):
        try:
            support = selector.get_support()
            final_feature_names = [name for name, kept in zip(symptoms_list, support) if kept]
        except:
            pass

    symptoms_dropdown = ["None"] + symptoms_list

    # ------------------- UI INPUTS ------------------------
    with st.expander("**Symptom Checker** (Click to expand/collapse)", expanded=True):
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
        X = v.reshape(1, -1)
        if scaler is not None: X = scaler.transform(X)
        if selector is not None: X = selector.transform(X)
        if pca is not None: X = pca.transform(X)
        return X

    st.write("")

    # ==========================================================
    #                   PREDICTION + XAI
    # ==========================================================
    if st.button("Analyze Health Condition"):

        if all(s == "None" for s in selected):
            st.error("⚠️ Please select at least one symptom to analyze.")
            return

        try:
            processed = preprocess(input_vec)

            # -------------------------------------------------
            # 1. PREDICTION
            # -------------------------------------------------
            try:
                proba = model.predict_proba(processed)[0]
            except:
                proba = np.array([1.0])

            pred_raw = model.predict(processed)[0]

            if label_encoder:
                pred_label = label_encoder.inverse_transform([int(pred_raw)])[0]
            else:
                pred_label = f"Class {pred_raw}"

            # ------------------- RESULT BOX -------------------
            st.markdown(f"""
            <div class="result-box">
                <h2>🧬 Diagnosis Result</h2>
                <h3 style="color:#00FFFF;">{pred_label}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Confidence Score", f"{np.max(proba)*100:.1f}%")

            st.write("---")
            st.subheader("Why did the AI make this prediction?")
            st.caption("The charts below show which symptoms contributed most.")

            # ==================================================
            #                   SHAP EXPLAINER
            # ==================================================
            with st.spinner("Running Explainable AI (SHAP)..."):
                
                background = np.zeros_like(processed)

                if hasattr(model, "predict_proba"):
                    pred_fn = model.predict_proba
                else:
                    pred_fn = model.predict

                explainer = shap.KernelExplainer(pred_fn, background)
                shap_vals = explainer.shap_values(processed, nsamples=100)

                target_idx = int(pred_raw)

                # Extract class SHAP values
                if isinstance(shap_vals, list):
                    shap_class = shap_vals[target_idx]
                else:
                    shap_class = shap_vals[0]

                shap_class = np.array(shap_class).reshape(-1)

                # Choose names
                current_names = final_feature_names if len(final_feature_names) == len(shap_class) else [f"Feature {i}" for i in range(len(shap_class))]

                explanation = shap.Explanation(
                    values=shap_class,
                    base_values=0,
                    data=processed[0],
                    feature_names=current_names
                )

                # ------------------- PLOTS -------------------
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### SHAP Waterfall Plot")
                    fig, ax = plt.subplots(figsize=(7,5))
                    shap.plots.waterfall(explanation, show=False, max_display=10)
                    st.pyplot(fig)
                    plt.clf()

                with col2:
                    st.markdown("### Top Factors")
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
            st.error(f"Analysis Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    st.set_page_config(page_title="Disease Predictor", layout="wide")
    general_disease_page()
