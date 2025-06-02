import faiss, numpy as np, asyncio

from utils import  batch_embed_text, langchain_chunk



class Faiss():
    
    def __init__(self): 
        self.index = None
        self.embedded_info = [] # This will store the list of dicts {id, values, metadata}
        self.dimension = None # To store the dimension of embeddings
          
    async def _get_embed_vals(self,client, texts:str, embed_method : str):
        
        chunks: list[str] = await langchain_chunk(texts, 300, 20)
        embedded = await batch_embed_text(client, chunks, method = embed_method)
        
        embed_mat = np.zeros((len(chunks), len(embedded[0]['values'])))  # assuming all have same dim

        for i, doc in enumerate(embedded):
            embed_mat[i] = np.array(doc['values'])

        return embed_mat, embedded
  
 
    # Faiss only allows upserting embed values 
    
    async def upsert_doc(self, client, texts:str, embed_method : str = "RETRIEVAL_DOCUMENT") -> faiss.IndexFlatL2:
        
        matrix_to_add, new_embedded_info = await self._get_embed_vals(client, texts, embed_method)

        if self.index is None:
            # First time upserting: initialize the index
            self.dimension = matrix_to_add.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.dimension != matrix_to_add.shape[1]:
            raise ValueError("Embedding dimension mismatch! Cannot add vectors of different dimensions to the same index.")

        self.index.add(matrix_to_add) # Add new vectors to the existing index
        self.embedded_info.extend(new_embedded_info) # Extend the list of metadata dictionaries


            
    async def query(self, client, *,  texts:str, embed_method : str = 'RETRIEVAL_QUERY', top_k:int = 3) -> list[str] | str:
        
        if self.index is None or not self.embedded_info:
            return "No documents indexed yet. Please upload a document first."

        queried = await batch_embed_text(client, texts, method = embed_method)
        query_matrix = np.array([queried[0]['values']]).astype('float32') # Ensure float32 for FAISS

        # Check query dimension against index dimension
        if query_matrix.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_matrix.shape[1]} does not match index dimension {self.dimension}")

        _, q = self.index.search(query_matrix, top_k) # comparing query embeds directly to the embeds in the index, then return a index with q
        
        
        
        # embed_info[q[0][0]], so q[0][0] returns a number which we use to index our list of embedded dicts, then we filter out the text from the meta data for the q index.
        results = [self.embedded_info[idx]['metadata']['text'] for idx in q[0]]
        
        return results
        
