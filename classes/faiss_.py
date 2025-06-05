import faiss, numpy as np

from utils import  batch_embed_text, langchain_chunk



class Faiss():
    
    def __init__(self): 
        self.index = None
        self.embedded_info = [] # This will store the list of dicts {id, values, metadata}
        self.dimension = None # To store the dimension of embeddings
          
    async def _get_embed_vals(self,client, texts:str, username:str,  embed_method : str):
        
        chunks: list[str] = await langchain_chunk(texts, 300, 20)
        embedded = await batch_embed_text(client, chunks, method = embed_method, id = username)
        
        embed_mat = np.zeros((len(chunks), len(embedded[0]['values'])))  # assuming all have same dim

        for i, doc in enumerate(embedded):
            embed_mat[i] = np.array(doc['values'])

        return embed_mat, embedded
  
 
    # Faiss only allows upserting embed values 
    
    async def upsert_doc(self, client, texts:str, username:str, embed_method : str = "RETRIEVAL_DOCUMENT") -> faiss.IndexFlatL2:
        
        matrix_to_add, new_embedded_info = await self._get_embed_vals(client, texts= texts, username = username,
                                                                      embed_method = embed_method)

        if self.index is None:
            # First time upserting: initialize the index
            self.dimension = matrix_to_add.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.dimension != matrix_to_add.shape[1]:
            raise ValueError("Embedding dimension mismatch! Cannot add vectors of different dimensions to the same index.")

        self.index.add(matrix_to_add) # Add new vectors to the existing index
        self.embedded_info.extend(new_embedded_info) # Extend the list of metadata dictionaries

        
    async def query(self, client, *, texts: str, user_id: str, embed_method: str = 'RETRIEVAL_QUERY', top_k: int = 3) -> list[str] | str:
            """
            Queries the FAISS index with the given text and always filters results by the provided user_id.

            """
            if self.index is None or not self.embedded_info:
                return "No documents indexed yet. Please upload a document first."

            queried = await batch_embed_text(client, texts, method=embed_method)
            query_matrix = np.array([queried[0]['values']]).astype('float32') # Ensure float32 for FAISS

            # Check query dimension against index dimension
            if query_matrix.shape[1] != self.dimension:
                raise ValueError(f"Query embedding dimension {query_matrix.shape[1]} does not match index dimension {self.dimension}")

            # Search for a larger number of candidates initially to increase the chances per user
            search_candidates_k = top_k * 2
            if search_candidates_k > len(self.embedded_info):
                search_candidates_k = len(self.embedded_info) # Don't search more than available docs

            _, distances_and_indices = self.index.search(query_matrix, search_candidates_k)
            
            # Filter results by user_id
            filtered_results_text = []
            for idx in distances_and_indices[0]:
                if idx >= len(self.embedded_info):
                    continue

                item_metadata = self.embedded_info[idx]['metadata']
                
                # Filer by id
                if item_metadata.get('user_id') == user_id:
                    filtered_results_text.append(item_metadata.get('text', '')) 
                    if len(filtered_results_text) >= top_k: # Stop once we have enough results
                        break
            
            return filtered_results_text
