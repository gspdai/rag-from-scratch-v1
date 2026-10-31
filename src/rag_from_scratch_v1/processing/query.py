import logging
import time

from rag_from_scratch_v1.guardrails import mask_pii

from .answerer import Answerer
from .vectordb import VectorDB
from .vectorizer import Vectorizer

logger = logging.getLogger(__name__)

class Query():
    def __init__(self, vectorizer: Vectorizer, db: VectorDB, llm: Answerer):
        self.vectorizer = vectorizer
        self.db = db
        self.llm = llm

    def ask(self, query:str, doc:str, top_k: int) -> str:
        start_time = time.perf_counter()
        mask_query = mask_pii(query)
        query_vec = self.vectorizer.query(text=query)
        top_matches = self.db.search(doc=doc,query_embed=query_vec,top_k=top_k)
        chunk_similarity = [(match.chunk.id,round(match.similarity,2)) for match in top_matches]
        response = self.llm.ask(query=query, top_matches=top_matches)
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = (block.text)
                break
        mask_response = mask_pii(response_text)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Query={mask_query!r} Response={mask_response!r} chunk={chunk_similarity} time_taken={total_time} input_tokens={response.usage.input_tokens} output_tokens={response.usage.output_tokens}")
        return response_text