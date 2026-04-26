from classes import Faiss
from utils import groq_generate, parse, transcribe
import asyncio
import streamlit as st
import base64
from google import genai
from groq import Groq
import hashlib 
import tempfile
from pathlib import Path


# --- API Key Setup ---
GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
client = genai.Client(api_key=GOOGLE_API_KEY)

GROQ_API_KEY = st.secrets["general"]["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Session State Initialization ---
# This variable will control whether the username has been entered
if 'username' not in st.session_state:
    st.session_state.username = None



# --- Username Input Screen ---
if st.session_state.username is None:
    st.title("Welcome to Chandra, Your Personalized AI Assistant!")
    st.write("Please enter your username to access your documents and chat.")

    with st.form("username_form"):
        username_input = st.text_input("Your Username", key="username_text_input")
        password_input = st.text_input("Enter Password Code", type="password")
        submit_button = st.form_submit_button("Enter")

        if submit_button:
            if not username_input or not password_input:
                st.warning("Both fields are required.")
            elif password_input.strip() != "qDEv" :
                st.error("Incorrect password. Please try again.")
            else:
                st.session_state.username = username_input
                st.success(f"Welcome, {st.session_state.username}!")
                st.rerun()
else:
    # --- Main Application (displayed after username is entered) ---
    st.title(f"Hello, {st.session_state.username}! Let's talk about your documents.")

    # --- Initialize Objects (can be cached if they don't change state much) ---

    @st.cache_resource
    def get_faiss_agent():
        return Faiss()
    
    faiss_agent = get_faiss_agent()
    # 
    # faiss_agent.index. #i wanted to see if it'll have similarity_search,(unless you loaded it from LangChain)

        
    # Initialize the chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": 'user', "content": 'Hello!'},
            {"role": 'AI assistant', "content": 'Hello, I am Chandra, your very own AI assistant for document understanding'},
            {"role": 'AI assistant', "content": 'How may I help you today'}
        ]


    def track_queries(query: str, previous_interactions: list[str]) -> str:
        
        interactions= '\n\n'.join(previous_interactions)
        
        augment_prompt = f'''\nPrevious interactions: {interactions}\nCurrent Query: {query}'''
        return augment_prompt

    # --- FILE UPLOADING AND INDEXING ---

    if 'processed_file_metadata' not in st.session_state:
        st.session_state.processed_file_metadata = {}


    uploaded_files = st.file_uploader(
        "Upload multiple documents",
        type=['txt', 'pdf', 'docx', 'pptx'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
    #     selected_file = st.selectbox("Select a file:", file_names, key="file_selector")
        
    if uploaded_files:
        files_to_process = []
        current_uploaded_file_names = {f.name for f in uploaded_files}
        to_append = list(current_uploaded_file_names)
        file_names.append(to_append)
        
        
        for uploaded_file in uploaded_files:
            
            uploaded_file.seek(0) 
            file_bytes = uploaded_file.read()
            file_hash = hashlib.md5(file_bytes).hexdigest() 

            # we use this to determine what files to upsert in session, by filtering those already uploaded and new ones
            if uploaded_file.name not in st.session_state.processed_file_metadata or \
            st.session_state.processed_file_metadata[uploaded_file.name] != file_hash:
                files_to_process.append((uploaded_file.name, file_bytes, file_hash))
                

        if files_to_process: # if we have a list of files now
            with st.spinner("Processing new/changed files... This may take a moment."):
                processed_count = 0
                for file_name, file_bytes, file_hash in files_to_process:
  
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir) / file_name
                        tmp_path.write_bytes(file_bytes)
                        try:
                            parsed_content = asyncio.run(parse(tmp_path))
                            
                            # your upsert_doc has to be able to extend/add to an index
                            asyncio.run(faiss_agent.upsert_doc(texts=f'NEW BOOK: {parsed_content}', 
                                                            username = st.session_state.username,
                                                            metadata= {"file_name": file_name})) # Pass doc_id
                            
                            st.session_state.processed_file_metadata[file_name] = file_hash # here is where we create a new hash
                            processed_count += 1
                        except Exception as e:
                            st.error(f"Error processing {file_name}: {e}")

            if processed_count > 0:
                st.success(f"✅ {processed_count} files processed!")

        elif uploaded_files and not files_to_process:
            st.info("All uploaded files are already processed.")


    # --- CONVERSATION FLOW (remains largely the same) ---



    # Display all previous messages (including the newly added ones from the last run)
    for message in st.session_state.messages[2:] : # Start from 2 to skip initial greetings (3 messages)
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'sources' in message:
                with st.expander("📚 View Source Context"):
                    for item in message['sources']:
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                            <div style="color: #0f52ba; font-weight: bold;">📄 {item.get('file_name', 'Unknown')}</div>
                            <div style="font-size: 0.9em;">{item.get('text', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            
    # --- UNIFIED CHAT INPUT (Robust DOM Injection) ---
    st.markdown("""
        <style>
        /* Hide default sidebar voice prompt if any */
        [data-testid="stSidebar"] .stAudioInput { display: none; }
        
        /* Make the audio input look like a button inside the flex container */
        [data-testid="stChatInput"] [data-testid="stAudioInput"] {
            width: 45px !important;
            height: 45px;
            overflow: hidden;
            background-color: transparent;
            border-radius: 50%;
            margin-left: 10px;
            transition: width 0.3s ease, background-color 0.3s ease;
        }
        
        /* Expand on hover/active */
        [data-testid="stChatInput"] [data-testid="stAudioInput"]:hover,
        [data-testid="stChatInput"] [data-testid="stAudioInput"]:focus-within {
            width: 250px !important;
            background-color: #f0f2f6;
            border-radius: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Invisible JS component to safely move the DOM elements
    import streamlit.components.v1 as components
    components.html("""
        <script>
        // Use a small delay to ensure Streamlit has rendered the elements
        setTimeout(function() {
            const audioInput = parent.document.querySelector('[data-testid="stAudioInput"]');
            const chatInput = parent.document.querySelector('[data-testid="stChatInput"]');
            
            // If both exist, and audio input is not already inside the chat input
            if (audioInput && chatInput && !chatInput.contains(audioInput)) {
                // Find the wrapper inside the chat input (where the textarea and send button live)
                const chatWrapper = chatInput.querySelector('div');
                if (chatWrapper) {
                    chatWrapper.appendChild(audioInput);
                } else {
                    chatInput.appendChild(audioInput);
                }
            }
        }, 500);
        </script>
    """, height=0, width=0)

    # Render elements normally, JS will move them
    audio_file = st.audio_input("Voice", label_visibility="collapsed")
    user_input = st.chat_input("Enter your prompt")

    if audio_file:
        with st.spinner(""):
            audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode()
            transcribed_text = asyncio.run(transcribe(base64_audio))
            if transcribed_text:
                user_input = transcribed_text

    if user_input:
        # here i am well not in a graceful way adding past interactions into my query...there was a way to add history to the llm
        query = track_queries(query = user_input, 
                            previous_interactions= [f"{msg['role']}: {msg['content']}" 
                                                    for msg in st.session_state.messages[-7:]] )
        # The new user message is displayed immediately
        with st.chat_message('user'):
            st.markdown(user_input)

        # The assistant's response is generated and streamed
        with st.chat_message('assistant'):

            extracted = asyncio.run(faiss_agent.query(texts=user_input, user_id=st.session_state.username,
                                                      top_k= 5))
            print(extracted)
            extracted_text = [i.get('text', '') for i in extracted] if isinstance(extracted, list) else None

            full_response = st.write_stream(groq_generate(query= query, relevant_passage=extracted_text))
            
            st.markdown("""
                <style>
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
            if extracted:
                with st.expander("View Source Context"):
                    for i, item in enumerate(extracted):
                        file_name = item.get("file_name", "Unknown File") 
                        text = item.get("text", "")
            
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-title">📄 {file_name}</div>
                            <div class="source-text">{text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
        # append when done
        st.session_state.messages.append({"role": 'user', "content": f'{user_input}'})
        if not extracted:
            st.session_state.messages.append({"role": 'AI assistant', "content": f'{full_response}'})
        else:
            st.session_state.messages.append({"role": 'AI assistant', "content": f'{full_response}',
                "sources": extracted})
                                        


