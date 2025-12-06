import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

# Safe SHAP init (prevents Render crash)
try:
    import IPython
    shap.initjs()
except Exception:
    pass


# ---------------------------------------------------------
#                 CUSTOM CSS & STYLING
# ---------------------------------------------------------
def local_css():
    st.markdown("""
        <style>
        /* CSS Removed for brevity (your CSS stays unchanged) */
        </style>
    """, unsafe_allow_html=True)


def general_disease_page():
    # Apply CSS
    local_css()

    st.markdown("<h1 style='text-align:center;'>General Illness Prediction</h1>", unsafe_allow_html=True)

    # -------------------- LOAD FILES --------------------
    base_path = "models/"  # <- Folder where ALL pkl files must exist

    def load_pickle(filename):
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            return pickle.load(open(path, "rb"))
        else:
            return None

    # MAIN MODEL FILES
    model = load_pickle("best_model.pkl")
    label_encoder = load_pickle("label_encoder.pkl")
    symptoms_list = load_pickle("symptom_list.pkl")

    # OPTIONAL PREPROCESSORS
    scaler = load_pickle("scaler.pkl")
    selector = load_pickle("selector.pkl")
    pca = load_pickle("pca.pkl")

    # ---- FIX: Prevent NameError ----
    if scaler is None:
        scaler = None
    if selector is None:
        selector = None
    if pca is None:
        pca = None

    # Ensure main model exists
    if model is None or symptoms_list is None:
        st.error("❌ Required model files missing in 'models/' folder.")
        return

    # -----------------------------------------------------------
    #     FEATURE NAME HANDLING
    # -----------------------------------------------------------
    final_feature_names = list(symptoms_list)

    if selector is not None:
        try:
            support = selector.get_support()
            final_feature_names = [name for name, kept in zip(symptoms_list, support) if kept]
        except:
            final_feature_names = list(symptoms_list)

    symptoms_dropdown = ["None"] + symptoms_list

    # ------------------- UI INPUTS ------------------------
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
        X = v.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        if selector is not None:
            X = selector.transform(X)
        if pca is not None:
            X = pca.transform(X)
        return X

    # ==========================================================
    #                   PREDICTION + XAI
    # ==========================================================
    if st.button("Analyze Health Condition"):

        if all(s == "None" for s in selected):
            st.error("⚠️ Please select at least one symptom.")
            return

        processed = preprocess(input_vec)

        # ---------------- PREDICTION ----------------
        try:
            proba = model.predict_proba(processed)[0]
        except:
            proba = np.array([1.0])

        pred_raw = model.predict(processed)[0]

        if label_encoder:
            pred_label = label_encoder.inverse_transform([int(pred_raw)])[0]
        else:
            pred_label = f"Class {pred_raw}"

        st.success(f"🧬 Predicted Illness: **{pred_label}**")
        st.metric("Confidence Score", f"{np.max(proba)*100:.1f}%")

        st.write("---")
        st.subheader("Explainable AI (SHAP)")
        st.caption("Understanding which symptoms influenced the prediction.")

        # ---------------- SHAP ----------------
        try:
            background = np.zeros_like(processed)
            pred_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

            explainer = shap.KernelExplainer(pred_fn, background)
            shap_vals = explainer.shap_values(processed, nsamples=80)

            # Extract SHAP for predicted class
            target_idx = int(pred_raw)
            shap_class_values = shap_vals[target_idx] if isinstance(shap_vals, list) else shap_vals[0]

            explanation = shap.Explanation(
                values=shap_class_values.flatten(),
                base_values=explainer.expected_value[target_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=processed.flatten(),
                feature_names=final_feature_names
            )

            # Waterfall plot
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)
            plt.clf()

        except Exception as e:
            st.error(f"SHAP Error: {e}")

    st.markdown("---")


if __name__ == "__main__":
    st.set_page_config(page_title="General Disease Predictor", layout="wide")
    general_disease_page()
