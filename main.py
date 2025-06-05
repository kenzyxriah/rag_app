from classes import ChatBot, Faiss
from utils import Parser
import asyncio
import streamlit as st
from google import genai
from groq import Groq
import hashlib 


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
        submit_button = st.form_submit_button("Enter")

        if submit_button:
            if username_input:
                st.session_state.username = username_input
                st.success(f"Welcome, {st.session_state.username}!")
                st.rerun() 
            else:
                st.warning("Username cannot be empty. Please enter a username.")
else:
    # --- Main Application (displayed after username is entered) ---
    st.title(f"Hello, {st.session_state.username}! Let's talk about your documents.")

    # --- Initialize Objects (can be cached if they don't change state much) ---
    parse_obj = Parser()

    @st.cache_resource
    def get_faiss_agent():
        return Faiss()

    faiss_agent = get_faiss_agent()

    bot = ChatBot()

        
    # Initialize the chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": 'user', "content": 'Hello!'},
            {"role": 'AI assistant', "content": 'Hello, I am Chandra, your very own AI assistant for document understanding'},
            {"role": 'AI assistant', "content": 'How may I help you today'}
        ]


    def track_queries(query: str, previous_interactions: list[str]) -> str:
        
        interactions= '\n\n'.join(previous_interactions)
        
        augment_prompt = f'''
        Previous interactions:
        {interactions}
        
        Current Query:
        {query}
        
        '''
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
        selected_file = st.selectbox("Select a file:", file_names, key="file_selector")
        
    if uploaded_files:
        files_to_process = []
        current_uploaded_file_names = {f.name for f in uploaded_files}
        to_append = list(current_uploaded_file_names)
        file_names.append(to_append)
        
        
        for uploaded_file in uploaded_files:
            
            uploaded_file.seek(0) 
            file_bytes = uploaded_file.read()
            file_hash = hashlib.md5(file_bytes).hexdigest() 

            # we use this to determine what files to upsert in session
            if uploaded_file.name not in st.session_state.processed_file_metadata or \
            st.session_state.processed_file_metadata[uploaded_file.name] != file_hash:
                files_to_process.append((uploaded_file.name, file_bytes, file_hash))
                

        if files_to_process:
            with st.spinner("Processing new/changed files... This may take a moment."):
                processed_count = 0
                for file_name, file_bytes, file_hash in files_to_process:
                    file_type = file_name.split('.')[-1].lower()
                    try:
                        parsed_content = asyncio.run(parse_obj.parse(file_bytes, ext=file_type))
                        
                        # your upsert_doc has to be able to extend/add to an index
                        asyncio.run(faiss_agent.upsert_doc(client, texts=f'NEW BOOK: {parsed_content}', 
                                                           username = st.session_state.username)) # Pass doc_id
                        
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
    for message in st.session_state.messages[2:] : # Start from 2 to skip initial greetings
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            
    # Get user input
    user_input = st.chat_input("Ask something:")

    if user_input:
        query = track_queries(query = user_input, 
                            previous_interactions= [f"{msg['role']}: {msg['content']}" 
                                                    for msg in st.session_state.messages[-7:]] )
        # The new user message is displayed immediately
        with st.chat_message('user'):
            st.markdown(user_input)

        # The assistant's response is generated and streamed
        with st.chat_message('assistant'):

            extracted = asyncio.run(faiss_agent.query(client, texts=user_input, user_id=st.session_state.username,
                                                      top_k= 5))
            
            # here i am well not in a graceful way adding past interactions into my query...there was a way to add history to the llm
            # this was just easier to understand 
            full_response_generator = bot.get_streaming_response(groq_client, input_statement= query, context=extracted)
            full_response = st.write_stream(full_response_generator)
                
        # append when done
        st.session_state.messages.append({"role": 'user', "content": f'{user_input}'})
        st.session_state.messages.append({"role": 'AI assistant', "content": f'{full_response}'})
                                        


