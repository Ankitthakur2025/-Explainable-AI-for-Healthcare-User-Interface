import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
st.markdown("""
<style>

/* FIX OPACITY ISSUE (Streamlit reduces metric opacity) */
[data-testid="stMetricValue"] > div,
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricValue"] span,
[data-testid="stMetricLabel"] span {
    color: black !important;
    opacity: 1 !important;          /* <— THIS IS THE REAL FIX */
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* REMOVE brightness, opacity, blur from metric container */
div[data-testid="metric-container"] {
    filter: none !important;
    opacity: 1 !important;
}

/* Also fix parent blocks (important!) */
div[data-testid="metric-container"] * {
    filter: none !important;
    opacity: 1 !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)




# --- Load model ---
try:
    heart_disease_model = pickle.load(open(r"D:\Research work\Mplementation\Web app\models\heart_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file not found. Please check the path and ensure the model is available.")
    # Dummy fallback model
    class DummyModel:
        def predict(self, df):
            if 'age' in df.columns and df['age'].iloc[0] > 55 and df['thalach'].iloc[0] < 120:
                return [1]
            return [0]

        def predict_proba(self, df):
            if 'age' in df.columns and df['age'].iloc[0] > 55 and df['thalach'].iloc[0] < 120:
                return np.array([[0.25, 0.75]])
            return np.array([[0.85, 0.15]])
    heart_disease_model = DummyModel()


# --- Helper for SHAP display ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# --- Donut Chart ---
def create_donut_chart(percentage, label, has_disease=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    if has_disease:
        main_color = '#d9534f'
        secondary_color = '#a94442'
    else:
        main_color = '#5cb85c'
        secondary_color = '#3e8e41'

    size_of_chunks = [percentage, 100 - percentage]
    ax.pie(size_of_chunks, colors=[main_color, secondary_color],
           wedgeprops=dict(width=0.3), startangle=90)
    centre_circle = plt.Circle((0, 0), 0.70, fc='black')
    fig.gca().add_artist(centre_circle)
    ax.text(0, 0, f"{percentage:.2f}%", ha='center', va='center',
            fontsize=22, color='white', weight='bold')
    fig.suptitle(label, fontsize=16, color='white', y=0.98,
                 backgroundcolor=main_color, x=0.5, ha='center', weight='bold')
    ax.axis('equal')
    return fig


# --- Main Page ---
def heart_disease_page():
    st.header("Heart Disease Prediction")

    col1, col2, col3, col4 = st.columns(4)
    with col1: age = st.number_input("Age", 1, 120, 50)
    with col2: sex = st.selectbox("Sex", ["Select", "Male", "Female"])
    with col3: cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
    with col1: trestbps = st.number_input("Resting BP", 80, 220)
    with col2: chol = st.number_input("Cholesterol (mg/dl)", 100, 600)
    with col3: fbs = st.number_input("Fasting Sugar >120 (1=True,0=False)", 0, 1)
    with col1: restecg = st.number_input("Resting ECG (0,1,2)", 0, 2)
    with col2: thalach = st.number_input("Max Heart Rate", 60, 220)
    with col3: exang = st.number_input("Exercise Angina (1=Yes,0=No)", 0, 1)
    with col1: oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0, 0.1)
    with col2: slope = st.number_input("Slope (0,1,2)", 0, 2)
    with col3: ca = st.number_input("Major Vessels (0-3)", 0, 3)
    with col1: thal = st.number_input("Thalassemia (1,2,3)", 1, 3)

    with col4:
        try:
            st.image(Image.open(r"image\heart.jpg"), width=300)
        except:
            st.image("https://via.placeholder.com/300x150.png?text=Heart", width=300)

    if st.button("Analyze Heart Health"):
        try:
            if sex == "Select":
                st.error("Please select a valid gender.")
                return

            sex_numeric = 1 if sex == "Male" else 0
            input_df = pd.DataFrame([[age, sex_numeric, cp, trestbps, chol, fbs, restecg,
                                      thalach, exang, oldpeak, slope, ca, thal]],
                                    columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                             'thalach','exang','oldpeak','slope','ca','thal'])

            prediction = heart_disease_model.predict(input_df)[0]
            proba = heart_disease_model.predict_proba(input_df)[0]

            st.subheader("Prediction Result")
            col_donut, col_metrics = st.columns([1, 2])
            with col_donut:
                if prediction == 1:
                    donut = create_donut_chart(proba[1]*100, "Heart Disease", True)
                else:
                    donut = create_donut_chart(proba[0]*100, "No Heart Disease", False)
                st.pyplot(donut)
                plt.clf()
                

            with col_metrics:
                st.metric("No Disease Probability", f"{proba[0]*100:.2f}%")
                st.metric("Disease Probability", f"{proba[1]*100:.2f}%")

            st.markdown("---")
            st.subheader("Explainable AI (SHAP) Visualization")

            # --- SHAP Explanation ---
            explainer = shap.TreeExplainer(heart_disease_model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                shap_array = np.array(shap_values[1]).reshape(1, -1)
                expected_value = explainer.expected_value[1]
            else:
                shap_array = np.array(shap_values).reshape(1, -1)
                expected_value = explainer.expected_value

            num_features = len(input_df.columns)
            safe_values = shap_array.flatten()[:num_features]

            if isinstance(expected_value, (list, np.ndarray)):
                base_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]
            else:
                base_value = float(expected_value)

            shap_explanation = shap.Explanation(
                values=safe_values,
                base_values=base_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns.tolist()
            )

            st.session_state["shap_explanation"] = shap_explanation
            st.session_state["prediction"] = prediction

            # --- Waterfall Plot ---
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(fig)
            plt.clf()

            # --- NEW CHART: Horizontal bar chart for Disease vs No Disease features ---
            st.markdown("<h3 style='color:#FFD700;'>Top Factors Affecting Disease vs No Disease</h3>", unsafe_allow_html=True)

            shap_df = pd.DataFrame({
                'Feature': shap_explanation.feature_names,
                'Importance': np.abs(shap_explanation.values),
                'SHAP Value': shap_explanation.values
            }).sort_values(by='Importance', ascending=False)

            top_pos = shap_df[shap_df['SHAP Value'] > 0].head(5)
            top_neg = shap_df[shap_df['SHAP Value'] < 0].head(5)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_pos['Feature'], top_pos['SHAP Value'], color='#FF3366', label='Increases Disease Risk')
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
            # --- END NEW CHART ---

            # --- Text Summary ---
            st.subheader("How Features Influence the Prediction")

            exp = st.session_state["shap_explanation"]
            prediction_label = 'Heart Disease' if st.session_state["prediction"] == 1 else 'No Heart Disease'
            
            shap_df = pd.DataFrame({
                'Feature': exp.feature_names,
                'SHAP Value': exp.values,
                'Feature Value': exp.data
            })
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values(by='Abs SHAP', ascending=False)
            
            st.markdown(f"The model predicted **{prediction_label}**. Here's a breakdown of the most influential factors:")

            positive_impact_features = shap_df[shap_df['SHAP Value'] > 0]
            if not positive_impact_features.empty:
                st.markdown("#### Features Increasing the Likelihood of Heart Disease:")
                for _, row in positive_impact_features.head(3).iterrows():
                    st.write(f"- **{row['Feature']}** (Value: `{row['Feature Value']}`): This was a significant factor pushing the prediction towards disease.")

            negative_impact_features = shap_df[shap_df['SHAP Value'] < 0]
            if not negative_impact_features.empty:
                st.markdown("#### Features Decreasing the Likelihood of Heart Disease:")
                for _, row in negative_impact_features.head(3).iterrows():
                    st.write(f"- **{row['Feature']}** (Value: `{row['Feature Value']}`): This factor pushed the prediction away from disease.")

        except Exception as e:
            st.error(f"An error occurred. Please ensure all inputs are valid numbers. Error: {e}")


# --- Run the Page ---
if __name__ == "__main__":
    heart_disease_page()
