import streamlit as st
from helper import predict
import os

# --- Page Config ---
st.set_page_config(
    page_title="DentAware - Vehicle Damage Classifier",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        /* Centering main title */
        .main-title {
            font-size: 2em;
            text-align: center;
            font-weight: bold;
            color: whitesmoke;
            margin-top: 1rem;
        }

        /* Subtitle slogans */
        .slogan {
            text-align: center;
            font-size: 1.2em;
            color: #555;
            font-style: italic;
            margin-bottom: 0.5rem;
        }


        /* Mobile responsive */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2em;
            }
            .slogan {
                font-size: 1em;
            }
        }

        footer {
            visibility: hidden;
        }

        /* Spinner override */
        .stSpinner > div > div {
            border-top-color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title & Slogans ---
st.markdown('<div class="main-title">🚙 DentAware - AI Powered Vehicle Damage Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">"Spot damage with precision \n, Protect your vehicle with AI eyes "</div>', unsafe_allow_html=True)

# --- File Uploader UI ---
with st.container():
    st.markdown('<div class="uploadbox">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📤 Upload a vehicle image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image showing the vehicle",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Logic ---
if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analyzing image... Please wait."):
            image_path = "temp_file.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            prediction = predict(image_path)

            if prediction:
                damage_type = prediction.get('class', 'Unknown')
                confidence = round(prediction.get('confidence', 0.0), 2)

                st.markdown(f"""
                    <div style="
                        border: 2px solid #28a745;
                        padding: 1.2rem;
                        border-radius: 10px;
                        background-color: whitesmoke;
                        color: #212121;
                        font-family: 'Segoe UI', sans-serif;
                    ">
                        <h4 style="color: #28a745; margin-bottom: 1rem;">✅ Damage Detected</h4>
                        <p style="margin: 0.5rem 0;"><strong style="color: #000;">Damage Type:</strong> {damage_type}</p>
                        <p style="margin: 0.5rem 0;"><strong style="color: #000;">Confidence:</strong> {confidence}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Could not analyze the image.")

            # Cleanup
            if os.path.exists(image_path):
                os.remove(image_path)

# --- Footer ---
st.markdown("---")
st.markdown("### 🔎 About")
st.markdown("DentAware is an AI-powered system that uses deep learning to detect and classify vehicle damage from images, identifying both the type and location with a certain level of confidence.")
st.markdown("**Built for accuracy, speed, and reliability — powered by AI for smarter, faster vehicle damage assessment. 🚘**")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <h4>A Project by ~ Adin Raja ✌️</h4>
    <p>
        <a href='https://github.com/adin11' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/adin-raja-492a78194/' target='_blank'>LinkedIn</a> | 
        <a href='mailto:adinraja78@gmail.com'>Email</a>
    </p>
    <small>© 2025 Adin Raja. All rights reserved.</small>
</div>
""", unsafe_allow_html=True)