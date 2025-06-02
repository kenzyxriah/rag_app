import asyncio

from utils import groq_generate



class ChatBot:
    class ChatBotException(Exception):
        pass

    @staticmethod
    async def process_input(client, context:str, user_input:str):
        # result = await gemini_generate(client, relevant_passage = context, query = user_input)
        result = groq_generate(client, relevant_passage = context, query = user_input)
        return result 
    
    def __init__(self):
        pass


    async def get_response(self, client, input_statement: str, context: str):
        result = await ChatBot.process_input(client, user_input= input_statement, context = context)
        return result
    
    
    async def get_streaming_response(self, client, input_statement: str, context: str):
        
        yield "🐱‍🏍Chandra is thinking...\n\n"
        
        # Get the result
        result = await ChatBot.process_input(client, user_input= input_statement, context = context)
        
        # Stream the actual response
        words = result.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
            await asyncio.sleep(0.07) # not too necessary
            
    
            
        
        



