import streamlit as st

# --- Helper Functions ---

def calculate_bmi(weight_kg, height_cm):
    """Calculates BMI from weight (kg) and height (cm)."""
    if height_cm == 0:
        return 0
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi

def interpret_bmi(bmi):
    """Returns the BMI category and a corresponding color."""
    if bmi < 18.5:
        return "Underweight", "#3498db"  # Blue
    elif 18.5 <= bmi < 25:
        return "Normal", "#2ecc71"  # Green
    elif 25 <= bmi < 30:
        return "Overweight", "#f1c40f"  # Yellow-Orange
    else:
        # Matches "Obese" range 30+
        return "Obese", "#e74c3c"  # Red

def calculate_bmi_prime(bmi):
    """Calculates BMI Prime (ratio of BMI to upper 'Normal' limit)."""
    return bmi / 25  # Using 25 as the upper limit from the image

def calculate_healthy_weight(height_cm):
    """Calculates the healthy weight range based on height."""
    height_m = height_cm / 100
    # Using 18.5 and 25 as the healthy BMI range from the image
    min_weight_kg = 18.5 * (height_m ** 2)
    max_weight_kg = 25 * (height_m ** 2)
    return min_weight_kg, max_weight_kg

def convert_us_to_metric(feet, inches, pounds):
    """Converts US units to metric units."""
    total_inches = (feet * 12) + inches
    height_cm = total_inches * 2.54
    weight_kg = pounds / 2.20462
    return height_cm, weight_kg

# --- Main App Page ---

def bmi_calculator_page():
    # st.set_page_config() IS REMOVED FROM HERE
    
    st.title("BMI CALCULATOR")

    # We use session state to store the results
    if 'bmi_result' not in st.session_state:
        st.session_state.bmi_result = None

    # Create two columns: one for inputs, one for results
    col1, col2 = st.columns([1, 1])

    with col1:
        # Use tabs for US and Metric units
        tab1, tab2 = st.tabs(["US Units", "Metric Units"])

        with tab1:
            # Inputs for US Units
            age_us = st.number_input("Age", min_value=2, max_value=120, value=25, key="age_us")
            gender_us = st.radio("Gender", ["Male", "Female"], index=0, horizontal=True, key="gender_us")
            
            st.write("Height")
            h_col1, h_col2 = st.columns(2)
            feet = h_col1.number_input("feet", min_value=0, value=5, step=1)
            inches = h_col2.number_input("inches", min_value=0, max_value=11, value=10, step=1)
            
            weight_lbs = st.number_input("Weight", min_value=0.0, value=160.0, step=0.1, help="pounds (lbs)")

            if st.button("Calculate BMI (US)", use_container_width=True):
                height_cm, weight_kg = convert_us_to_metric(feet, inches, weight_lbs)
                st.session_state.bmi_result = {
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "unit_system": "US"
                }

        with tab2:
            # Inputs for Metric Units
            age_metric = st.number_input("Age", min_value=2, max_value=120, value=25, key="age_metric")
            gender_metric = st.radio("Gender", ["Male", "Female"], index=0, horizontal=True, key="gender_metric")
            
            height_cm_metric = st.number_input("Height", min_value=0.0, value=170.0, step=0.1, help="centimeters (cm)")
            weight_kg_metric = st.number_input("Weight", min_value=0.0, value=70.0, step=0.1, help="kilograms (kg)")

            if st.button("Calculate BMI (Metric)", use_container_width=True):
                st.session_state.bmi_result = {
                    "height_cm": height_cm_metric,
                    "weight_kg": weight_kg_metric,
                    "unit_system": "Metric"
                }
    
    with col2:
        st.header("Result")

        if st.session_state.bmi_result:
            height_cm = st.session_state.bmi_result["height_cm"]
            weight_kg = st.session_state.bmi_result["weight_kg"]
            unit_system = st.session_state.bmi_result["unit_system"]

            bmi = calculate_bmi(weight_kg, height_cm)
            category, color = interpret_bmi(bmi)
            bmi_prime = calculate_bmi_prime(bmi)
            min_w_kg, max_w_kg = calculate_healthy_weight(height_cm)

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3>Your BMI is</h3>
                    <h1 style="font-size: 4em; margin: 0; color: {color};">{bmi:.1f}</h1>
                    <h2 style="color: {color};">({category})</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.divider()
            st.subheader("Details")
            st.markdown(f"**Healthy BMI range:** 18.5 kg/m² - 25 kg/m²")

            if unit_system == "US":
                min_w_lbs = min_w_kg * 2.20462
                max_w_lbs = max_w_kg * 2.20462
                st.markdown(f"**Healthy weight for the height:** {min_w_lbs:.1f} lbs - {max_w_lbs:.1f} lbs")
            else:
                st.markdown(f"**Healthy weight for the height:** {min_w_kg:.1f} kg - {max_w_kg:.1f} kg")

            st.markdown(f"**BMI Prime:** {bmi_prime:.2f}")
            
            height_m = height_cm / 100
            if height_m > 0:
                ponderal_index = weight_kg / (height_m ** 3)
                st.markdown(f"**Ponderal Index:** {ponderal_index:.1f} kg/m³")
        else:
            st.info("Enter your details in the left panel and click 'Calculate BMI' to see your results here.")

# This part is fine, it allows you to run this file by itself for testing
if __name__ == "__main__":
    bmi_calculator_page()