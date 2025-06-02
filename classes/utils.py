from google import genai
from google.genai import types
import uuid, asyncio
import streamlit as st, re

from groq import Groq


GROQ_API_KEY =st.secrets["general"]["GROQ_API_KEY"]
groq_client = Groq(api_key= GROQ_API_KEY)


GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]

client = genai.Client(api_key= GOOGLE_API_KEY)


def embedder(client, model:str, content:str|list[str], method:str):
    
    return client.models.embed_content(
    model=model,
    contents=content,
    config=types.EmbedContentConfig(task_type=method))
    
    


        
    
def groq_generate(client, query: str, relevant_passage: str|list[str], max_tokens: int=4096) -> str:

    context_and_history = '\n\n'.join(relevant_passage) if not isinstance(relevant_passage, str) else relevant_passage
    
    PROMPT_TEMPLATE = f"""
            You are a helpful and informative bot assistant that answers questions using
            text from the reference passage included below. Be sure to respond in a complete sentence, being comprehensive,
            including all relevant background information. 
            
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
            
            Context + History - {context_and_history}
            
            QUESTION: '{query}'      

        Kindly provide the answer.
        """
    
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant chatbot"},
                {"role": "user", "content": PROMPT_TEMPLATE}
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content.strip()
        processed_text = re.split(r"</think>\s*", text, maxsplit=1)[-1]
        
        return processed_text
    except Exception as e:
        raise RuntimeError(f"Error generating completion: {e}") from e
    
    
import tqdm


async def batch_embed_text(client, texts: str|list[str], batch_size=10, *, method: str = 'semantic_similarity', 
                     model_name="models/text-embedding-004", metadata: dict = {}, id=None):
    
    config = []
    
    # in this case.. one string, make it a list to be looped through
    if isinstance(texts, str):
        texts = [texts]



    # If len(texts) = 1 and batch_size = 5 (for example)
    # range(0, 1, 5) produces [0], So batch_start will be 0 (just one iteration)
    
    for batch_start in tqdm.tqdm(range(0, len(texts), batch_size)):
    
        size = min(len(texts) - batch_start, batch_size)
        batch_texts = texts[batch_start:batch_start+size]

        # embed batch
        response = embedder(client, model = model_name, content = batch_texts, method = method)
        
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
