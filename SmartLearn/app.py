import streamlit as st

st.set_page_config(
    page_title="SmartLearn",
    layout="wide"
)

# Custom CSS for "Nude" Color Palette and Modern Design
st.markdown("""
<style>
    /* Main Background and Text Colors */
    .stApp {
        background-color: #FdfcF0; /* Cream/Off-white */
        color: #4A4A4A; /* Dark Grey specifically for text */
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2C2C2C !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #D4C4B7; /* Latte/Beige */
        color: #2C2C2C;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #C3B0A0; /* Darker Beige */
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Cards/Containers (if any specific styling needed) */
    .css-1r6slb0 {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F5F1E8; /* Light Beige */
    }

    /* Hero Section Styling */
    .hero-container {
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #F5F1E8 0%, #FFFFFF 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #8E7F75, #4A4A4A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'user' not in st.session_state:
    st.session_state.user = "Guest"

# Welcome Effect (Varied based on user feedback - removed)
if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = True

def main_dashboard():
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Welcome to SmartLearn</div>
            <div class="hero-subtitle">Your Intelligent Companion for Mastering New Skills</div>
        </div>
    """, unsafe_allow_html=True)

    # Content Grid
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("###  **AI Chatbot**")
        st.info("Interactive tutoring session with our advanced AI assistant. Ask questions, get explanations, and learn at your own pace.")
        if st.button("Start Chatting ->"):
            st.switch_page("pages/Chatbot.py")

    with col2:
        st.markdown("### **Exercise Generator**")
        st.success("Practice makes perfect. Generate custom exercises, quizzes, and problems tailored to your current level and subject.")
        if st.button("Generate Exercises ->"):
            st.switch_page("pages/Exercise_Generator.py")
    
    st.markdown("---")
    st.markdown("#### **Getting Started is Easy**")
    st.markdown("""
    1.  **Select a Tool**: Choose between the Chatbot or Exercise Generator from the cards above or the sidebar.
    2.  **Interact**: Type your questions or specify your exercise parameters.
    3.  **Learn**: Receive instant feedback and detailed explanations.
    """)

# Main Entry Point
main_dashboard()
