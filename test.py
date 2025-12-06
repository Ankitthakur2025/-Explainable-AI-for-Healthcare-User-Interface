import streamlit as st
import base64
from pathlib import Path

# ⚠️ DO NOT uncomment set_page_config if it's already used elsewhere
# st.set_page_config(page_title="Explainable AI for Healthcare", layout="wide")

# --- Function to convert local GIF to base64 ---
def get_base64_gif(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        data_url = base64.b64encode(data).decode("utf-8")
        return f"data:image/gif;base64,{data_url}"
    except Exception as e:
        st.error(f"Error loading GIF: {e}")
        return None

def home_page():
    
    gif_path = Path(r"image/online-doctor-giving-advice-help-laptop.png")

    # Load the GIF in base64 format
    gif_data = get_base64_gif(gif_path)

    # --- Custom Styling ---
    st.markdown("""
<style>

    /* --- Light Premium Background --- */
    .stApp {
        background: #f4faff !important;
        font-family: 'Poppins', sans-serif;
    }

    /* --- Navbar --- */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 50px;
        background: #ffffff;
        border-bottom: 1px solid #d9ecf2;
        border-radius: 0 0 20px 20px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    }

    .nav-left {
        font-size: 30px;
        font-weight: 700;
        color: #009b84;
        text-shadow: none;
    }

    /* --- Hero Section --- */
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 60px 70px;
        animation: fadeIn 1s ease-in-out;
    }

    /* --- Hero Text --- */
    .hero-text {
        max-width: 55%;
    }

    .hero-text h1 {
        font-size: 50px;
        font-weight: 800;
        color: #007a64;
        margin-bottom: 20px;
    }

    .hero-text p {
        font-size: 20px;
        color: #355c58;
        margin-bottom: 25px;
        line-height: 1.5;
    }

    /* --- Premium Button --- */
    .hero-btn {
        background: linear-gradient(90deg, #00c9a9, #009b7a);
        color: white;
        padding: 12px 35px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0px 4px 14px rgba(0,150,120,0.25);
        transition: 0.2s ease-in-out;
    }

    .hero-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0px 6px 20px rgba(0,150,120,0.35);
    }

    /* --- Hero Image --- */
    .hero-img img {
        width: 360px;
        border-radius: 20px;
        box-shadow: 0px 4px 18px rgba(0,150,120,0.12);
        animation: fadeInUp 1.2s ease-in-out;
    }

    /* --- Animations --- */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }

</style>
""", unsafe_allow_html=True)


    # --- Navbar ---
    st.markdown("""
    <div class="navbar">
        <div class="nav-left">Explainable AI for Healthcare</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Hero Section ---
    if gif_data:
        hero_html = f"""
        <div class="hero">
            <div class="hero-text">
                <h1>Welcome to the Smart Healthcare System </h1>
                <p>Experience the future of healthcare with our Explainable AI System that predicts and explains risks for 
                <b>Heart Disease</b> , <b>Diabetes</b> and <b>General Disease</b>. — combining accuracy with transparency.</p>
                <a href="#" class="hero-btn">Explore Now</a>
            </div>
            <div class="hero-img">
                <img src="{gif_data}" alt="Medical AI Illustration">
            </div>
        </div>
        """
    else:
        hero_html = """
        <div class="hero">
            <div class="hero-text">
                <h1>AI-Driven Disease Prediction</h1>
                <p>Welcome to Explainable AI for Healthcare — combining accuracy, transparency, and trust in medical predictions.</p>
                <a href="#" class="hero-btn">Explore Now</a>
            </div>
            <div class="hero-img">
                <p style='color:#66fcf1;'>⚠️ GIF not found. Please check file path.</p>
            </div>
        </div>
        """

    st.markdown(hero_html, unsafe_allow_html=True)


if __name__ == "__main__":
    home_page()
