import streamlit as st
import streamlit.components.v1 as components
import base64
import asyncio
from .utils import transcribe

def inject_chat_css():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] .stAudioInput,
        [data-testid="stAudioInput"] { 
            position: fixed !important;
            bottom: -1000px; 
        }
        
        .voice-active {
            bottom: 38px !important;
            display: block !important;
            width: 45px !important;
            height: 45px;
            overflow: hidden;
            background-color: transparent;
            border-radius: 25px;
            z-index: 1000000;
            transition: width 0.3s ease;
        }
        
        .voice-active:hover, .voice-active:focus-within {
            width: 250px !important;
            background-color: #f0f2f6;
        }

        [data-testid="stChatInput"] textarea {
            padding-right: 55px !important;
        }

        .source-box {
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .source-title {
            font-weight: bold;
            color: #0f52ba;
            margin-bottom: 5px;
        }
        .source-text {
            font-size: 0.9em;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

def inject_chat_js():
    components.html("""
        <script>
        function sync() {
            const audioInput = parent.document.querySelector('[data-testid="stAudioInput"]');
            const chatInput = parent.document.querySelector('[data-testid="stChatInput"]');
            
            if (audioInput && chatInput) {
                const rect = chatInput.getBoundingClientRect();
                audioInput.classList.add('voice-active');
                audioInput.style.left = (rect.right - 55) + 'px';
                audioInput.style.bottom = (window.innerHeight - rect.bottom + 8) + 'px';
            }
            const stray = parent.document.querySelectorAll('[data-testid="stChatInput"] [data-testid="stAudioInput"]');
            stray.forEach(el => el.remove());
        }
        setInterval(sync, 100);
        </script>
    """, height=0, width=0)

def render_unified_input():
    if 'audio_key' not in st.session_state:
        st.session_state.audio_key = 0
    
    inject_chat_css()
    inject_chat_js()
    
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
