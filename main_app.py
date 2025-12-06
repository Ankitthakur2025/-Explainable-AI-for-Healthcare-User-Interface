import os, sys
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

st.markdown("""
<style>

    /* Global background */
    .stApp {
        background-color: #f7fafc !important;
        font-family: 'Segoe UI', sans-serif;
        color: black !important;
        text-align: left !important;
    }

    /* Fix main content container width & alignment */
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        text-align: left !important;
    }

    /* Sidebar - borderless */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0d3b2e !important;
        font-size: 16px;
    }

    /* Active menu - borderless */
    .nav-link.active {
        background-color: #8be3b7 !important;
        color: #003322 !important;
        border-radius: 8px;
        font-weight: 700;
        border: none !important;
    }
    .nav-link:hover {
        background-color: #c8f7e5 !important;
        color: #00664d !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #008060 !important;
        font-weight: 700;
        text-align: left !important;
    }

    /* Cards / containers — remove borders */
    .stMarkdown, .stNumberInput, .stTextInput, .stSelectbox, .stTextArea, .stRadio, .stDateInput {
        background-color: #ffffff00 !important;
        padding: 8px !important;
        border: none !important;
        box-shadow: none !important;
        text-align: left !important;
    }

    /* Input fields */
    input, textarea, select {
        background-color: white !important;
        border: none !important;
        border-radius: 6px !important;
        color: #003322 !important;
        box-shadow: none !important;
        text-align: left !important;
    }

    input:focus, textarea:focus, select:focus {
        border: none !important;
        outline: none !important;
        box-shadow: 0 0 0 2px #00b38666 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00b386, #009970);
        color: white !important;
        padding: 0.6rem 1.2rem;
        border: none !important;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
        box-shadow: 0 4px 12px rgba(0,150,100,0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,150,100,0.3);
    }

    /* Result Box */
    .result-box {
        background: #eafff5;
        padding: 20px;
        border: none !important;
        border-radius: 10px;
        font-size: 18px;
        color: black;
        box-shadow: none !important;
        text-align: left !important;
    }

    /* Input labels */
    label,
    .stTextInput label, 
    .stNumberInput label,
    .stSelectbox label,
    .stMultiselect label,
    .stSlider label,
    .stRadio label,
    .stDateInput label,
    .stTextArea label {
        color: #008060 !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        opacity: 1 !important;
        border: none !important;
        text-align: left !important;
    }

    /* Placeholder text */
    input::placeholder, textarea::placeholder {
        color: #00a78c !important;
        opacity: 0.6 !important;
    }

    /* Dropdown options */
    .css-11unzgr, .css-1n7v3ny {
        background: white !important;
        color: #003d33 !important;
        border: none !important;
    }

    /* Selectbox internal border remove */
    div[data-baseweb="select"] {
        border: none !important;
        box-shadow: none !important;
    }

    /* Number Input spinner */
    button.step-up, button.step-down {
        background-color: #006f5a !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }

    /* Remove Streamlit default borders */
    .css-1y4p8pa, .css-1cpxqw2, .css-1n76uvr, .css-1d2x8ny {
        border: none !important;
        box-shadow: none !important;
    }

    /* ⭐ METRIC TEXT FIX */
    [data-testid="stMetricValue"], 
    [data-testid="stMetricValue"] *, 
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricLabel"] * {
        color: black !important;
        opacity: 1 !important;
        filter: none !important;
    }

    /* ⭐ FIX TABS (US / METRIC) */
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f0ff !important;
        border-radius: 6px !important;
        padding: 4px 12px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #d0e2ff !important;
        color: #0047b3 !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab"] * {
        color: #0047b3 !important;
        opacity: 1 !important;
        filter: none !important;
    }

    /* ⭐ FINAL FIX — MAKE GENDER (Male/Female) TEXT VISIBLE */
    div[role="radiogroup"] p,
    div[role="radiogroup"] label,
    div[role="radio"] p,
    div[role="radio"] span,
    div[role="radio"] * {
        color: #000 !important;
        opacity: 1 !important;
        filter: none !important;
    }

</style>
""", unsafe_allow_html=True)


# Allow Python to find local modules
sys.path.append(os.path.dirname(__file__))

# Import pages
from heart_disease import heart_disease_page
from diabetes import diabetes_page
from general_disease import general_disease_page
from test import home_page
from bmi_calculator import bmi_calculator_page

# Sidebar navigation
with st.sidebar:
    try:
        image = Image.open("image/logo.png")
        st.image(image, width=300)
    except FileNotFoundError:
        st.image(
            "https://via.placeholder.com/300x150.png?text=Logo",
            caption="Placeholder Logo",
            width=300
        )

    selected = option_menu(
        "Explainable AI System",
        ["Home", "Heart Disease", "Diabetes", "General Diseases", "BMI CALCULATOR"],
        icons=['house', 'heart', 'activity', 'bandaid', 'clipboard-data'],
        default_index=0
    )

# Render correct page
if selected == "Home":
    home_page()
elif selected == "Heart Disease":
    heart_disease_page()
elif selected == "Diabetes":
    diabetes_page()
elif selected == "General Diseases":
    general_disease_page()
elif selected == "BMI CALCULATOR":
    bmi_calculator_page()
