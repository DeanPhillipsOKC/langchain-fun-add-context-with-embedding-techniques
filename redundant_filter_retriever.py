from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document

class RedundantFilterRetriever(BaseRetriever):
    # Whenever somebody creates this class they will have to provide embeddings.  This
    # helps keep us from hard coding it to open ai embeddings.
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        # Need to define this, but I'm not going to use it for this exercise.
        return []