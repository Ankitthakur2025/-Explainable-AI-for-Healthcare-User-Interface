import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

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
            background-color: #FFFFFF !important;      /* FIX BLACK BOX */
            padding: 20px !important;
            border-radius: 12px !important;
            text-align: center !important;
            box-shadow: 0px 4px 10px rgba(0,255,255,0.2);
            color: #000 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #000000 !important;                 /* FIX TEXT */
            font-size: 2.8rem !important;
            font-weight: 800 !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #004d39 !important;                 /* FIX LABEL */
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

        /* Results Container — FIX BLACK BOX */
        .result-box {
            background-color: #FFFFFF !important;      /* FIX BLACK BOX */
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #00FFFF;
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
            color: #000 !important;
        }

        /* Fix text inside result box */
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

    # -------------------- LOAD FILES --------------------
    # UPDATE PATHS AS NECESSARY FOR YOUR LOCAL MACHINE
    base_path = "models/"
    
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


    if not model or not symptoms_list:
        st.error(f"Critical files missing. Please check paths. Looking in: {base_path}")
        return

    # -----------------------------------------------------------
    #     FIX: UPDATE SYMPTOM NAMES IF FEATURE SELECTION USED
    # -----------------------------------------------------------
    # If a selector (like SelectKBest) removed features, our 'symptoms_list'
    # needs to shrink to match the model's expected inputs for SHAP labels.
    final_feature_names = list(symptoms_list)
    if selector is not None:
        try:
            # Get boolean mask of selected features
            support = selector.get_support()
            # Filter the names list
            final_feature_names = [name for name, kept in zip(symptoms_list, support) if kept]
        except Exception as e:
            # Fallback if selector is not standard sklearn
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
            except ValueError:
                pass 

    # ------------------- PREPROCESS -------------------
    def preprocess(v):
        X = v.reshape(1, -1)
        if scaler is not None: X = scaler.transform(X)
        if selector is not None: X = selector.transform(X)
        # Note: If PCA is used, SHAP will explain PCA components, not symptoms.
        # Mapping back from PCA to symptoms is complex and inexact.
        if pca is not None: X = pca.transform(X)
        return X

    st.write("") # Spacer

    # ==========================================================
    #                   PREDICTION + XAI
    # ==========================================================
    if st.button("Analyze Health Condition"):

        if all(s == "None" for s in selected):
            st.error("⚠️ Please select at least one symptom to analyze.")
            return

        try:
            processed = preprocess(input_vec)

            # -----------------------------------------
            # 1. PREDICTION
            # -----------------------------------------
            try:
                proba = model.predict_proba(processed)[0]
                is_multiclass = (len(proba) > 1)
            except:
                proba = np.array([1.0])
                is_multiclass = False

            pred_raw = model.predict(processed)[0]

            if label_encoder:
                idx_to_decode = int(pred_raw)
                pred_label = label_encoder.inverse_transform([idx_to_decode])[0]
            else:
                pred_label = f"Class {pred_raw}"

            # ------------------- RESULT DISPLAY -------------------
            st.markdown(f"""
            <div class="result-box">
                <h2 style="margin-top:0; color:#FFFFFF;">🧬 Diagnosis Result</h2>
                <h3 style="color:#00FFFF;">{pred_label}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1,1,1])
            with c2:
                st.metric("Confidence Score", f"{np.max(proba)*100:.1f}%")

            st.write("---")
            st.subheader("Why did the AI make this prediction?")
            st.caption("The charts below show which symptoms contributed most to this diagnosis.")

            with st.spinner("Running Explainable AI (SHAP)..."):
                # ==================================================
                #              ROBUST SHAP LOGIC
                # ==================================================
                background = np.zeros_like(processed) 
                
                if hasattr(model, "predict_proba"):
                    pred_fn = model.predict_proba
                else:
                    pred_fn = model.predict

                explainer = shap.KernelExplainer(pred_fn, background)
                shap_vals = explainer.shap_values(processed, nsamples=100)

                # -----------------------------------------
                # SAFE EXTRACT LOGIC
                # -----------------------------------------
                target_idx = int(pred_raw)
                shap_class_values = None
                base_val = 0.0

                def get_expected_value(exp_val, idx):
                    if isinstance(exp_val, (list, np.ndarray)):
                        if len(np.shape(exp_val)) > 0 and len(exp_val) > idx:
                            return float(exp_val[idx])
                        elif len(np.shape(exp_val)) > 0:
                            return float(exp_val[0])
                        else:
                            return float(exp_val)
                    return float(exp_val)

                if isinstance(shap_vals, list):
                    if target_idx < len(shap_vals):
                        shap_class_values = np.array(shap_vals[target_idx]).reshape(-1)
                        base_val = get_expected_value(explainer.expected_value, target_idx)
                    else:
                        shap_class_values = np.array(shap_vals[0]).reshape(-1)
                        base_val = get_expected_value(explainer.expected_value, 0)
                
                elif isinstance(shap_vals, np.ndarray):
                    if shap_vals.shape == (1, processed.shape[1]):
                        shap_class_values = shap_vals[0]
                        base_val = get_expected_value(explainer.expected_value, 0)
                    elif shap_vals.shape[0] > 1 and shap_vals.shape[1] == processed.shape[1]:
                        if target_idx < shap_vals.shape[0]:
                            shap_class_values = shap_vals[target_idx]
                            base_val = get_expected_value(explainer.expected_value, target_idx)
                        else:
                            shap_class_values = shap_vals[0]
                            base_val = get_expected_value(explainer.expected_value, 0)
                    elif len(shap_vals.shape) == 3:
                        if shap_vals.shape[2] > target_idx:
                            shap_class_values = shap_vals[0, :, target_idx]
                        else:
                            shap_class_values = shap_vals[0, :, 0]
                        base_val = get_expected_value(explainer.expected_value, target_idx)
                    elif shap_vals.size == processed.shape[1]:
                        shap_class_values = shap_vals.reshape(-1)
                        base_val = get_expected_value(explainer.expected_value, 0)

                if shap_class_values is not None:
                    data_1d = processed[0].reshape(-1)
                    values_1d = shap_class_values.reshape(-1)
                    
                    # -----------------------------------------
                    # FEATURE NAME MAPPING
                    # -----------------------------------------
                    # Use 'final_feature_names' calculated earlier (handles Selector)
                    # If PCA is active, names won't match, so we fallback to generic.
                    current_names = final_feature_names if len(final_feature_names) == len(values_1d) else [f"Feature {i}" for i in range(len(values_1d))]

                    explanation = shap.Explanation(
                        values=values_1d,
                        base_values=base_val,
                        data=data_1d,
                        feature_names=current_names
                    )

                    # -----------------------------------------
                    # PLOTS
                    # -----------------------------------------
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        st.markdown("##### Explainable AI (SHAP) Visualization")
                        fig, ax = plt.subplots(figsize=(8,6))
                        try:
                            shap.plots.waterfall(explanation, show=False, max_display=10)
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning("Standard plot unavailable, switching to bar view.")
                            shap.plots.bar(explanation, show=False, max_display=10)
                            st.pyplot(fig)
                        plt.clf()

                    with col_p2:
                        st.markdown("##### Top Factors Affecting Disease")
                        shap_df = pd.DataFrame({
                            "Feature": explanation.feature_names,
                            "SHAP Value": explanation.values,
                            "Importance": np.abs(explanation.values)
                        }).sort_values("Importance", ascending=False).head(10)

                        fig, ax = plt.subplots(figsize=(8,6))
                        # Cleaner Bar Chart
                        colors = ['#00FFFF' if x > 0 else '#FF4B4B' for x in shap_df["SHAP Value"]]
                        ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
                        ax.set_xlabel("Impact on Model Output")
                        ax.invert_yaxis()
                        ax.set_facecolor("#F8FAFE") # Match Streamlit dark theme if possible
                        fig.patch.set_facecolor('#0E1117')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['top'].set_color('none')
                        ax.spines['right'].set_color('none')
                        ax.spines['left'].set_color('white')
                        
                        st.pyplot(fig)
                        plt.clf()

                    st.write("---")
                    st.markdown("##### Visual Impact Analysis (Force Plot)")
                    try:
                        shap.plots.force(explanation, matplotlib=True, show=False)
                        fig_force = plt.gcf()
                        fig_force.set_size_inches(16, 4)
                        # Styling force plot bg
                        fig_force.patch.set_facecolor("#EDEFF3")
                        st.pyplot(fig_force)
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Force plot unavailable: {e}")

                else:
                    st.warning("Could not extract SHAP values.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            import traceback
            traceback.print_exc()

    st.markdown("---")

if __name__ == "__main__":
    st.set_page_config(page_title="Disease Predictor", layout="wide")
    general_disease_page()
