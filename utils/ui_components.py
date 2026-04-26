import streamlit as st
import streamlit.components.v1 as components
import base64
import asyncio
import textwrap
from .utils import transcribe

def apply_premium_theme():
    base_css = textwrap.dedent("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        .stApp, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 50% 0%, #3d2b1f 0%, #0d0e15 100%) !important;
            font-family: 'Outfit', sans-serif !important;
            color: #ffffff !important;
        }
        
        h1, h2, h3, h4, h5, h6, p, label {
            font-family: 'Outfit', sans-serif !important;
        }
 
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(15px) !important;
            border-radius: 20px !important;
            padding: 1.5rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }
 
        .stTextInput input, .stNumberInput input, .stSelectbox > div[data-baseweb="select"] {
            background: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 10px !important;
        }
        .stTextInput input:focus, .stSelectbox > div[data-baseweb="select"]:focus-within {
            border-color: #ff9d00 !important;
            box-shadow: 0 0 0 1px #ff9d00 !important;
        }
 
        .stButton button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: #ff9d00 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(255, 157, 0, 0.2) !important;
        }
        
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #ff9d00 0%, #f9d423 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
        }
        .stButton button[kind="primary"]:hover {
            opacity: 0.9 !important;
            box-shadow: 0 8px 24px rgba(255, 157, 0, 0.4) !important;
        }
        
        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] .stAudioInput,
        [data-testid="stAudioInput"] { 
            position: fixed !important;
            bottom: 57px !important;
            right: calc(50% - 340px) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 41px !important;
            height: 41px !important;
            overflow: hidden;
            background-color: transparent !important;
            border-radius: 20px !important;
            z-index: 1000000;
            transition: all 0.3s ease !important;
        }
        
        @media (max-width: 800px) {
            [data-testid="stAudioInput"] {
                right: 20px !important;
            }
        }

        [data-testid="stAudioInput"]:hover, [data-testid="stAudioInput"]:focus-within {
            width: 250px !important;
            margin-left: -205px !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
        }

        [data-testid="stBottomBlockContainer"] {
            background: transparent !important;
            border: none !important;
        }

        [data-testid="stChatInput"] {
            background: transparent !important;
        }

        [data-testid="stChatInput"] textarea {
            padding-right: 55px !important;
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }

        .source-box {
            margin-bottom: 10px;
            border: 1px solid rgba(255, 157, 0, 0.2);
            border-radius: 12px;
            padding: 12px;
            background-color: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(5px);
        }
        .source-title {
            font-weight: bold;
            color: #ff9d00;
            margin-bottom: 5px;
        }
        .source-text {
            font-size: 0.9em;
            color: #e5e7eb;
        }
    </style>
    """)
    st.markdown(base_css, unsafe_allow_html=True)

def render_unified_input():
    if 'audio_key' not in st.session_state:
        st.session_state.audio_key = 0
    
    audio_file = st.audio_input("Voice", label_visibility="collapsed", key=f"audio_prompt_{st.session_state.audio_key}")
    chat_input = st.chat_input("Enter your prompt")
    
    user_input = None
    if chat_input:
        user_input = chat_input
    elif 'temp_voice_input' in st.session_state:
        user_input = st.session_state.pop('temp_voice_input')
    elif audio_file:
        with st.spinner(""):
            audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode()
            transcribed_text = asyncio.run(transcribe(base64_audio))
            if transcribed_text:
                st.session_state.temp_voice_input = transcribed_text
                st.session_state.audio_key += 1
                st.rerun()
                
    return user_input
