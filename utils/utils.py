from google import genai
from google.genai import types
import uuid, asyncio
import streamlit as st, re, json
from googleapiclient.discovery import build
from traceback import print_exc
from groq import AsyncGroq
from datetime import datetime

import logging
import warnings

# Ignore all warnings for the rest of the process
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. Get a logger instance (you can name it or use the root logger)
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
            If the passage is irrelevant or not to helpful to the answer, 
            you may use ypur internal knowledge to respond, or ask the user to provide more context as what you have is not suitale
            
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
        
    
    
    
import tqdm

# this is google's client internally for embedding
async def batch_embed_text(texts: str|list[str], batch_size=10, *, method: str = 'semantic_similarity', 
                     model_name="models/gemini-embedding-001", metadata: dict = {}, id=None):
    
    '''
    Batch embedding texts using Gemini Models which can be selected by model_name
    '''
    
    config = []
    
    # in this case.. one string, make it a list to be looped through
    if isinstance(texts, str):
        texts = [texts]

    
    for batch_start in tqdm.tqdm(range(0, len(texts), batch_size)):
    
        size = min(len(texts) - batch_start, batch_size)
        batch_texts = texts[batch_start:batch_start+size]

        # embed batch
        response = embedder(model = model_name, content = batch_texts, method = method)
        
        # create config data for upserting for batch
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



from langchain_text_splitters import RecursiveCharacterTextSplitter

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



