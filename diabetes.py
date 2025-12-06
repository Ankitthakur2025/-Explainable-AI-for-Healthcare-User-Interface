
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------------- CONFIG (change these to your paths) ----------------
MODEL_PATH = r"D:\Research work\Mplementation\WEB APP\models\best_diabetes_model.pkl"
# If available, set the CSV used for training so SHAP can use a representative background
TRAINING_DATA_PATH = r"D:\Research work\Mplementation\WEB APP\diabetes_prediction_dataset.csv"

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    diabetes_model = load_model(MODEL_PATH)
except Exception as e:
    st.warning(f"Could not load model at {MODEL_PATH}. Using dummy fallback. Error: {e}")
    class DummyModel:
        def predict(self, X): 
            # expects dataframe
            try:
                g = float(X.iloc[0].get('blood_glucose_level', 120))
            except Exception:
                g = 120.0
            return np.array([1 if g > 140 else 0])
        def predict_proba(self, X):
            try:
                g = float(X.iloc[0].get('blood_glucose_level', 120))
            except Exception:
                g = 120.0
            if g > 140: return np.array([[0.05, 0.95]])
            elif g < 100: return np.array([[0.95, 0.05]])
            else: return np.array([[0.6, 0.4]])
    diabetes_model = DummyModel()

# If the loaded object is a pipeline, extract classifier & preprocessor
if hasattr(diabetes_model, "named_steps"):
    try:
        classifier = diabetes_model.named_steps.get("classifier", diabetes_model)
    except Exception:
        classifier = diabetes_model
    try:
        preprocessor = diabetes_model.named_steps.get("preprocessor", None)
    except Exception:
        preprocessor = None
else:
    classifier = diabetes_model
    preprocessor = None

# ---------- helper: build a SHAP background dataset (tries training csv then synthetic) ----------
def build_shap_background(preprocessor, train_csv_path=None, n_samples=200):
    # 1) try load provided training CSV (best)
    if train_csv_path and os.path.exists(train_csv_path):
        try:
            df = pd.read_csv(train_csv_path)
            df = df.dropna()
            if df.shape[0] == 0:
                raise ValueError("Training CSV had zero usable rows")
            df_sample = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)
            # If preprocessor exists, try to transform background to model input space
            if preprocessor is not None:
                try:
                    X_bg = preprocessor.transform(df_sample)
                    # get output feature names if available (sklearn >=1.0)
                    try:
                        cols = preprocessor.get_feature_names_out()
                    except Exception:
                        cols = [f"f{i}" for i in range(X_bg.shape[1])]
                    return pd.DataFrame(X_bg, columns=cols)
                except Exception:
                    # transformation failed — fall back to raw df_sample
                    return df_sample.reset_index(drop=True)
            else:
                return df_sample.reset_index(drop=True)
        except Exception:
            pass

    # 2) synthetic fallback background (small, plausible)
    rng = np.random.RandomState(42)
    n = min(100, n_samples)
    genders = ['Male', 'Female', 'Other']
    smokes = ['never', 'former', 'current', 'not current', 'ever', 'No Info']
    bg = pd.DataFrame({
        'gender': rng.choice(genders, size=n),
        'age': rng.randint(20, 70, size=n),
        'hypertension': rng.choice([0,1], size=n, p=[0.8,0.2]),
        'heart_disease': rng.choice([0,1], size=n, p=[0.9,0.1]),
        'smoking_history': rng.choice(smokes, size=n),
        'bmi': np.clip(rng.normal(26,4,size=n), 15, 50),
        'HbA1c_level': np.clip(rng.normal(5.8,1.2,size=n), 3, 15),
        'blood_glucose_level': np.clip(rng.normal(110,35,size=n), 50, 400)
    })
    if preprocessor is not None:
        try:
            X_bg = preprocessor.transform(bg)
            try:
                cols = preprocessor.get_feature_names_out()
            except Exception:
                cols = [f"f{i}" for i in range(X_bg.shape[1])]
            return pd.DataFrame(X_bg, columns=cols)
        except Exception:
            return bg
    else:
        return bg

# build once
SHAP_BACKGROUND = build_shap_background(preprocessor, TRAINING_DATA_PATH, n_samples=200)

# ---------- small ui helper ----------
def create_donut(percentage, label, is_disease):
    fig, ax = plt.subplots(figsize=(3,3))
    color_primary = '#FF3366' if is_disease else '#00FF99'
    ax.pie([percentage, 100-percentage], colors=[color_primary, '#333333'], startangle=90, wedgeprops=dict(width=0.3))
    ax.text(0, 0, f"{percentage:.1f}%", ha='center', va='center', fontsize=16, color=color_primary, fontweight='bold')
    ax.set_title(label, color='white', fontsize=12, pad=-5)
    fig.patch.set_facecolor("#0E1117")
    return fig

# --------------------- MAIN PAGE ---------------------
def diabetes_page():
    # Do not call st.set_page_config here if main_app already set it
    model_type = type(classifier).__name__
    st.markdown("<h1 style='margin-bottom:0;'>Diabetes Prediction System</h1>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 2px solid #00FFFF; border-radius: 5px;'>", unsafe_allow_html=True)

    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Select","Male","Female","Other"])
        heart_disease = st.selectbox("Heart Disease (History)", [0,1])
        HbA1c_level = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5, step=0.1)
        hypertension = st.selectbox("Hypertension (High BP)", [0,1])
    with c2:
        age = st.number_input("Age", 1, 120, 30)
        smoking_history = st.selectbox("Smoking History", ['never','former','current','not current','ever','No Info'])
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", 50, 400, 120)
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 70.0, 25.0, step=0.1)
    with c3:
        try:
            st.image(r"image\diabetes.png", width=250)
        except Exception:
            st.image("https://via.placeholder.com/250x150.png?text=Diabetes+Care", width=250)

    st.markdown("---")

    # Button
    if st.button("Analyze Diabetes Risk"):
        if gender == "Select":
            st.error("Please select a valid gender.")
            return

        # Build raw input df (strings for categorical)
        input_df = pd.DataFrame([{
            'gender': gender,
            'age': float(age),
            'hypertension': int(hypertension),
            'heart_disease': int(heart_disease),
            'smoking_history': smoking_history,
            'bmi': float(bmi),
            'HbA1c_level': float(HbA1c_level),
            'blood_glucose_level': float(blood_glucose_level)
        }])

        # Predict using pipeline or raw model (pipeline will handle encoding)
        try:
            pred = diabetes_model.predict(input_df)[0]
            proba = diabetes_model.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Results display
        
        st.subheader("Prediction Result")
        dcol, mcol = st.columns([1,2])
        with dcol:
            is_d = int(pred) == 1
            val = proba[1]*100 if is_d else proba[0]*100
            label = "Diabetes" if is_d else "No Diabetes"
            fig = create_donut(val, label, is_d)
            st.pyplot(fig)
        with mcol:
            st.metric("No Diabetes Probability", f"{proba[0]*100:.2f}%")
            st.metric("Diabetes Probability", f"{proba[1]*100:.2f}%")
            if int(pred) == 1:
                st.error("The model predicts that the person **is diabetic**.")
            else:
                st.success("The model predicts that the person **is not diabetic**.")

        # ========= SHAP Explanation block (user requested format) =========
        st.markdown("---")
        st.subheader("Explainable AI (SHAP)")

        try:
            # Prepare X_for_shap: must be the model's numeric input space
            if preprocessor is not None:
                try:
                    X_trans = preprocessor.transform(input_df)
                    # get feature names
                    try:
                        feat_names = preprocessor.get_feature_names_out()
                    except Exception:
                        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
                    X_for_shap = pd.DataFrame(X_trans, columns=feat_names)
                    # Prepare background in transformed space if possible
                    X_bg = None
                    if SHAP_BACKGROUND is not None:
                        try:
                            if isinstance(SHAP_BACKGROUND, pd.DataFrame) and SHAP_BACKGROUND.shape[1] == X_for_shap.shape[1]:
                                X_bg = SHAP_BACKGROUND
                            else:
                                # try to transform raw SHAP_BACKGROUND (if it was raw df)
                                if hasattr(SHAP_BACKGROUND, 'columns') and set(['gender','age']).issubset(set(SHAP_BACKGROUND.columns)):
                                    bg_trans = preprocessor.transform(SHAP_BACKGROUND)
                                    try:
                                        bg_cols = preprocessor.get_feature_names_out()
                                    except Exception:
                                        bg_cols = [f"f{i}" for i in range(bg_trans.shape[1])]
                                    X_bg = pd.DataFrame(bg_trans, columns=bg_cols)
                        except Exception:
                            X_bg = None
                except Exception as e:
                    # if preprocessing transform fails, try to use input_df raw (may still work for catboost)
                    X_for_shap = input_df.copy()
                    X_bg = SHAP_BACKGROUND
            else:
                X_for_shap = input_df.copy()
                X_bg = SHAP_BACKGROUND

            # Run TreeExplainer / Explainer with fallbacks
            shap_values = None
            used_method = None

            # If classifier is CatBoostClassifier, TreeExplainer often accepts raw DataFrame with categorical strings
            if isinstance(classifier, CatBoostClassifier):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_for_shap)
                used_method = "TreeExplainer (CatBoost)"
            else:
                # For trees (XGBoost) prefer tree_path_dependent with background
                if isinstance(classifier, XGBClassifier):
                    try:
                        if X_bg is not None:
                            explainer = shap.Explainer(classifier, X_bg, feature_perturbation="tree_path_dependent")
                            shap_values = explainer(X_for_shap)
                            used_method = "Explainer (XGB, tree_path_dependent w/bg)"
                        else:
                            # try tree_path_dependent without explicit background
                            explainer = shap.Explainer(classifier, feature_perturbation="tree_path_dependent")
                            shap_values = explainer(X_for_shap)
                            used_method = "Explainer (XGB, tree_path_dependent)"
                    except Exception:
                        # fallback to interventional
                        explainer = shap.Explainer(classifier, feature_perturbation="interventional")
                        shap_values = explainer(X_for_shap)
                        used_method = "Explainer (XGB, interventional fallback)"
                else:
                    # generic classifier (sklearn pipeline etc.)
                    try:
                        explainer = shap.Explainer(classifier, X_bg if X_bg is not None else None)
                        shap_values = explainer(X_for_shap)
                        used_method = f"Explainer ({type(classifier).__name__})"
                    except Exception:
                        # last resort: interventional
                        explainer = shap.Explainer(classifier, feature_perturbation="interventional")
                        shap_values = explainer(X_for_shap)
                        used_method = f"Explainer ({type(classifier).__name__}, interventional)"

            # convert shap_values to the original style used in your pasted block:
            # shap_values may be either an Explanation object or a numpy array/list for shap older versions
            if hasattr(shap_values, "values"):
                # shap.Explanation result
                # shap_values.values shape (n_samples, n_features) for single output, or (n_samples, n_classes, n_features)
                vals = shap_values.values
                # handle multi-output (probabilities) like classic shap_tree
                if vals.ndim == 3:
                    # multi-class -> take class 1 contribution if exists
                    shap_array = np.array(vals[0,1,:]).reshape(1, -1)
                    base_val = shap_values.base_values[0][1] if isinstance(shap_values.base_values, (list, np.ndarray)) and np.array(shap_values.base_values).ndim>1 else shap_values.base_values
                else:
                    shap_array = np.array(vals[0]).reshape(1, -1)
                    base_val = shap_values.base_values[0] if isinstance(shap_values.base_values, (list, np.ndarray)) else shap_values.base_values
            else:
                # legacy shap returns list/array (older API)
                if isinstance(shap_values, list):
                    shap_array = np.array(shap_values[1]).reshape(1, -1)
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    shap_array = np.array(shap_values).reshape(1, -1)
                    base_val = explainer.expected_value

            # prepare safe values limited to number of features of X_for_shap
            num_features = X_for_shap.shape[1] if hasattr(X_for_shap, 'shape') else len(shap_array.flatten())
            safe_values = shap_array.flatten()[:num_features]

            # compute base_value neatly
            if isinstance(base_val, (list, np.ndarray)):
                if len(np.atleast_1d(base_val)) > 1:
                    try:
                        base_value = float(base_val[1])
                    except Exception:
                        base_value = float(base_val[0])
                else:
                    base_value = float(np.atleast_1d(base_val)[0])
            else:
                base_value = float(base_val)

            # feature names: if we transformed, use those, else input_df columns
            feature_names = list(X_for_shap.columns) if hasattr(X_for_shap, "columns") else input_df.columns.tolist()

            # Build shap.Explanation like your block
            shap_explanation = shap.Explanation(
                values=safe_values,
                base_values=base_value,
                data=X_for_shap.iloc[0] if hasattr(X_for_shap, 'iloc') else input_df.iloc[0],
                feature_names=feature_names
            )

            # store in session_state (as your block expects)
            st.session_state["shap_explanation"] = shap_explanation
            st.session_state["prediction"] = int(pred)

            # --- Waterfall Plot ---
            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                shap.plots.waterfall(shap_explanation, show=False)
                st.pyplot(fig)
            except Exception:
                # fallback: horizontal bar of top features
                shap_df_plot = pd.DataFrame({
                    'Feature': shap_explanation.feature_names,
                    'SHAP Value': shap_explanation.values
                }).sort_values(by='SHAP Value', key=lambda s: np.abs(s), ascending=False).head(10)
                fig2, ax2 = plt.subplots(figsize=(10,6))
                colors = ['#FF3366' if v>0 else '#00FF99' for v in shap_df_plot['SHAP Value']]
                ax2.barh(shap_df_plot['Feature'], shap_df_plot['SHAP Value'], color=colors)
                ax2.invert_yaxis()
                st.pyplot(fig2)
            plt.clf()

            # --- NEW HORIZONTAL BAR: Disease vs No Disease features ---
            st.markdown("<h3 style='color:#FFD700;'>Top Factors Affecting Disease vs No Disease</h3>", unsafe_allow_html=True)
            shap_df = pd.DataFrame({
                'Feature': shap_explanation.feature_names,
                'Importance': np.abs(shap_explanation.values),
                'SHAP Value': shap_explanation.values
            }).sort_values(by='Importance', ascending=False)

            top_pos = shap_df[shap_df['SHAP Value'] > 0].head(5)
            top_neg = shap_df[shap_df['SHAP Value'] < 0].head(5)

            fig, ax = plt.subplots(figsize=(10, 5))
            if not top_pos.empty:
                ax.barh(top_pos['Feature'], top_pos['SHAP Value'], color='#FF3366', label='Increases Disease Risk')
            if not top_neg.empty:
                ax.barh(top_neg['Feature'], top_neg['SHAP Value'], color='#00FF99', label='Reduces Disease Risk')
            ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=12, color='white')
            ax.set_ylabel("Feature", fontsize=12, color='white')
            ax.set_title("Most Influential Features for Disease vs No Disease", fontsize=14, color='#FFD700', fontweight='bold')
            ax.legend(facecolor='#0E1117', labelcolor='white')
            ax.set_facecolor("#0E1117")
            fig.patch.set_facecolor("#0E1117")
            ax.tick_params(colors='white', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#FFD700')
            st.pyplot(fig)
            plt.clf()

            # --- Text Summary (exactly as you requested, adapted for diabetes) ---
            st.subheader("How Features Influence the Prediction")

            exp = st.session_state["shap_explanation"]
            prediction_label = 'Diabetes' if st.session_state["prediction"] == 1 else 'No Diabetes'

            shap_df_text = pd.DataFrame({
                'Feature': exp.feature_names,
                'SHAP Value': exp.values,
                'Feature Value': exp.data
            })
            shap_df_text['Abs SHAP'] = shap_df_text['SHAP Value'].abs()
            shap_df_text = shap_df_text.sort_values(by='Abs SHAP', ascending=False)

            st.markdown(f"The model predicted **{prediction_label}**. Here's a breakdown of the most influential factors:")

            positive_impact_features = shap_df_text[shap_df_text['SHAP Value'] > 0]
            if not positive_impact_features.empty:
                st.markdown("#### Features Increasing the Likelihood of Diabetes:")
                for _, row in positive_impact_features.head(3).iterrows():
                    st.write(f"- **{row['Feature']}** (Value: `{row['Feature Value']}`): This was a significant factor pushing the prediction towards disease.")

            negative_impact_features = shap_df_text[shap_df_text['SHAP Value'] < 0]
            if not negative_impact_features.empty:
                st.markdown("#### Features Decreasing the Likelihood of Diabetes:")
                for _, row in negative_impact_features.head(3).iterrows():
                    st.write(f"- **{row['Feature']}** (Value: `{row['Feature Value']}`): This factor pushed the prediction away from disease.")

        except Exception as e:
            st.error(f"An error occurred during SHAP visualization. Error: {e}")

        # BMI Insight
        st.markdown("---")
        st.subheader("⚖️ Health Insight: BMI Status")
        if bmi < 18.5:
            st.info("Underweight — consider consulting a nutritionist.")
        elif bmi < 25:
            st.success("Normal weight — great job!")
        elif bmi < 30:
            st.warning("Overweight — regular exercise recommended.")
        else:
            st.error("Obese — lifestyle modification advised.")

# Run page directly
if __name__ == "__main__":
    diabetes_page()
