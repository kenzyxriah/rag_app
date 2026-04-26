import hashlib 
import base64
import os
import uuid, asyncio
import streamlit as st, re, json
import tempfile
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from traceback import print_exc
from groq import AsyncGroq
from datetime import datetime
import logging
import warnings
import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

GROQ_API_KEY =st.secrets["general"]["GROQ_API_KEY"]
groq_client = AsyncGroq(api_key= GROQ_API_KEY)
cx = st.secrets["general"]["CX"] 
google_search_key = st.secrets["general"]["GOOGLE_SEARCH"]

GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
gemini_client = genai.Client(api_key= GOOGLE_API_KEY)

tools = [
        {
            "type": "browser_search"
        }
    ]

def embedder(model:str, content:str|list[str], method:str):
    return gemini_client.models.embed_content(
    model=model,
    contents=content,
    config=types.EmbedContentConfig(task_type=method, 
                                    output_dimensionality= 768))
        
async def groq_generate(query: str, relevant_passage: str|list[str] = None, max_tokens: int=4096):
    idx = query.index("Current Query:")
    logger.info(query[idx + 14: ].strip())

    context = ''
    if relevant_passage:
        context = '\n\n'.join(relevant_passage) if not isinstance(relevant_passage, str) else relevant_passage

    sys_instructions = f"""
            You are a helpful and informative bot assistant that answers questions using
            text from the reference passage included below. Be sure to respond in a complete sentence, being comprehensive,
            including all relevant background information. 
            You have a web search tool that you should use immediately it is required
            Here is the attached date of this conversation : {datetime.now()}
            
            Here is your work flow

            1. 
            If there is no passage attached or the passage is irrelevant and not helpful to the query, 
            you may use ypour internal knowledge to respond, or ask the user to provide more context as what you have is not suitale
            
            2. 
            History tabs are denoted by 
            - Role: User
            - Role: AI Assitant
            - Context: RAG
            
            Tailor responses taking user's past interaction into context in relevant passages provided to you
            
            
            3. Using previous interactions, you find the nuance of the current questions/queryextracted
            History + Current QUESTION:      

        Kindly provide the answer.
        """
    PROMPT_TEMPLATE = f"""
           {'Context' if context else ''} {context}
            QUESTION: '{query}'      
        """
    messages = [
                {"role": "system", "content": sys_instructions},
                {"role": "user", "content": PROMPT_TEMPLATE}]
    
    try:
        completion = await groq_client.chat.completions.create(
            model='openai/gpt-oss-20b',
            messages=messages,
            tools=tools,
            tool_choice='auto',
            temperature=0.5,
            reasoning_effort= "low",
            max_tokens=max_tokens,
            stream = True
        )

        async for chunk in completion:
            if data := chunk.choices[0].delta.content:
                processed_text = re.split(r"</think>\s*", data, maxsplit=1)[-1]
                yield processed_text

    except Exception as e:
        logger.error(str(e))
        raise RuntimeError(f"Error generating completion: {e}") from e

async def batch_embed_text(texts: str|list[str], batch_size=10, *, method: str = 'semantic_similarity', 
                     model_name="models/gemini-embedding-001", metadata: dict = {}, id=None):
    
    config = []
    
    # in this case.. one string, make it a list to be looped through
    if isinstance(texts, str):
        texts = [texts]

    for batch_start in tqdm.tqdm(range(0, len(texts), batch_size)):
        size = min(len(texts) - batch_start, batch_size)
        batch_texts = texts[batch_start:batch_start+size]
        response = embedder(model = model_name, content = batch_texts, method = method)
        data =  [
        {
                "id": str(uuid.uuid4()),
                "values": embedding.values,
                "metadata": {
                    "text": batch_texts[i], 
                    'user_id': id,
                    **(metadata)
                }
            }
        for i, embedding in enumerate(response.embeddings)
        ]
        config.extend(data)
    return config

async def langchain_chunk(text: str, size:int, overlap:int) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "", '.', "NEW BOOK: "],  
        is_separator_regex= True,    
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    clean_single_chunk = lambda x: x.replace('\n\n', '').replace('\n', '').replace('  ', '')
    chunks = await asyncio.gather(*[asyncio.to_thread(clean_single_chunk, chunk) for chunk in chunks])
    return chunks

async def save_audio_from_base64(base64_string: str, file_format: str = "wav"):
    base64_cleaned = base64_string.strip().replace("\n", "").replace("\r", "")
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp:
            output_file_path = tmp.name
            audio_bytes = base64.b64decode(base64_cleaned)
            tmp.write(audio_bytes)
        
        logger.info(f"Audio decoded and saved successfully to {output_file_path}")
        return output_file_path
    except Exception as e:
        logger.error(f"Failed to decode audio: {e}")
        return None

stt_prompt = """ 
You are a highly accurate AI transcription assistant. Your task is to convert spoken language into clean, readable text.

Guidelines:
- Transcribe the spoken input word-for-word, preserving the speaker’s intent.
- Correct minor grammatical issues only if necessary for clarity.
- Do not add or omit any information.
- Use punctuation to make the transcription easier to read (e.g., commas, periods).
- If the speaker hesitates or repeats a word, clean it up unless it changes the meaning.
- Avoid inserting labels like "um", "uh", unless they’re contextually important.

Context:
The following audio was captured as part of a user request. Your job is to return the most accurate, raw readable transcript possible.
"""

async def convert_audio_to_text(audio_file_path: str) -> str:
    try:
        with open(audio_file_path, "rb") as file:
            transcription = await groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                prompt=stt_prompt
            )
        return transcription
    except Exception as e:
        logger.error(f"Error during Groq transcription: {e}")
        return ""

async def transcribe(base64_audio: str):
    audio_path = await save_audio_from_base64(base64_audio)
    if not audio_path:
        return None
    try:
        transcript = await convert_audio_to_text(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return transcript
