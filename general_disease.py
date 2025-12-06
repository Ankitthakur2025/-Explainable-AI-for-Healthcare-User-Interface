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
        h1 {
            color: #00FFFF; 
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            text-shadow: 0px 0px 10px rgba(0, 255, 255, 0.3);
        }
        h3 {
            color: #E0E0E0;
        }

        /* Metric Cards */
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

        /* Button */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #008B8B 0%, #00FFFF 100%);
            color: #000000;
            font-weight: bold;
            border: none;
            padding: 0.8rem;
            border-radius: 10px;
        }

        .result-box {
            background-color: #FFFFFF !important;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #00FFFF;
            margin-top: 20px;
            margin-bottom: 20px;
            color: #000 !important;
        }
        </style>
    """, unsafe_allow_html=True)



# ==========================================================
#                 GENERAL DISEASE PAGE
# ==========================================================
def general_disease_page():

    local_css()

    st.markdown(
        "<h1 style='text-align:center;'>General Illness Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<p style='text-align:center; color:#AAAAAA;'>Select symptoms below.</p>", unsafe_allow_html=True)

    # ---------------------- LOAD FILES ----------------------
    def load_pickle(filename):
        if os.path.exists(filename):
            return pickle.load(open(filename, "rb"))
        else:
            return None

    model = load_pickle("best_model.pkl")
    label_encoder = load_pickle("label_encoder.pkl")
    symptoms_list = load_pickle("symptom_list.pkl")

    scaler = load_pickle("scaler.pkl")
    selector = load_pickle("selector.pkl")
    pca = load_pickle("pca.pkl")

    # ALWAYS define selector safely
    if selector is None:
        selector_available = False
    else:
        selector_available = True

    if symptoms_list is None:
        st.error("❌ Symptoms list not found. Upload symptom_list.pkl")
        return

    if model is None:
        st.error("❌ Model file missing. Upload best_model.pkl")
        return

    # --------------------- FEATURE NAMES SAFE LOGIC ---------------------
    final_feature_names = list(symptoms_list)

    if selector_available:
        try:
            support = selector.get_support()
            final_feature_names = [name for name, keep in zip(symptoms_list, support) if keep]
        except:
            pass

    symptoms_dropdown = ["None"] + symptoms_list

    # ----------------------- UI INPUTS ----------------------
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

    # ----------------------- INPUT VECTOR ----------------------
    input_vec = np.zeros(len(symptoms_list))

    for s in selected:
        if s != "None":
            if s in symptoms_list:
                idx = symptoms_list.index(s)
                input_vec[idx] = 1

    # ----------------------- PREPROCESS ----------------------
    def preprocess(v):
        X = v.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        if selector_available:
            X = selector.transform(X)
        if pca is not None:
            X = pca.transform(X)
        return X


    # ==========================================================
    #                      PREDICTION
    # ==========================================================
    if st.button("Analyze Health Condition"):

        if all(s == "None" for s in selected):
            st.error("⚠️ Select at least one symptom.")
            return

        processed = preprocess(input_vec)

        proba = model.predict_proba(processed)[0]
        pred_raw = model.predict(processed)[0]

        if label_encoder:
            pred_label = label_encoder.inverse_transform([int(pred_raw)])[0]
        else:
            pred_label = f"Class {pred_raw}"

        # ------------------- RESULT BOX -------------------
        st.markdown(f"""
        <div class="result-box">
            <h2>🧬 Diagnosis Result</h2>
            <h3 style="color:#00CCCC;">{pred_label}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence", f"{np.max(proba)*100:.1f}%")

        st.write("---")
        st.subheader("Explainable AI (SHAP)")

        # ==========================================================
        #               SHAP EXPLAINER (SAFE MODE)
        # ==========================================================
        try:
            background = np.zeros_like(processed)

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(processed, nsamples=80)

            target_idx = int(pred_raw)
            shap_values = shap_vals[target_idx][0]

            # MATCH FEATURE NAMES SAFELY
            if len(shap_values) == len(final_feature_names):
                feature_names = final_feature_names
            else:
                feature_names = [f"Feature {i}" for i in range(len(shap_values))]

            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value[target_idx],
                data=processed[0],
                feature_names=feature_names
            )

            # ---------------- WATERFALL ---------------
            st.markdown("#### 🔍 SHAP Waterfall Chart")
            fig, ax = plt.subplots(figsize=(8,6))
            shap.plots.waterfall(explanation, show=False, max_display=10)
            st.pyplot(fig)
            plt.clf()

            # ---------------- BAR PLOT ---------------
            st.markdown("#### 📊 Feature Importance (Top Factors)")
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP": shap_values,
                "ABS": np.abs(shap_values)
            }).sort_values("ABS", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(7,5))
            ax.barh(shap_df["Feature"], shap_df["SHAP"])
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.clf()

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        st.write("---")


# RUN PAGE
if __name__ == "__main__":
    st.set_page_config(page_title="Disease Predictor", layout="wide")
    general_disease_page()
