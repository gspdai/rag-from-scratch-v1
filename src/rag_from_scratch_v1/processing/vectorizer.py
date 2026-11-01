from dataclasses import dataclass

import voyageai
import mlflow

from .chunker import Chunk

from rag_from_scratch_v1.guardrails import mask_pii
from rag_from_scratch_v1.pricing import Price


@dataclass(frozen=True)
class Vector:
    chunk: Chunk
    embedding:tuple[float,...]

class Vectorizer():
    def __init__(self, client: voyageai.Client, model: str, price:Price = Price(input_token_cost_per_million=0)):
        self.client = client
        self.model = model
        self.model_price = price


    def doc(self, chunks:list[Chunk]) -> list[Vector]:
        chunk_text = [chunk.text for chunk in chunks]
        response = self.client.embed(texts=chunk_text, model=self.model, input_type="document")
        vector = [Vector(chunk=chunk, embedding=tuple(embedding)) for chunk, embedding in zip(chunks,response.embeddings)]
        return vector

    def query(self, text:str) -> tuple[float,...]:
        with mlflow.start_span(name="generate_embed") as span:
            mask_text = mask_pii(text=text)
            span.set_inputs({"query":mask_text})
            response = self.client.embed(texts= [text], model=self.model, input_type="query")
            query_cost = self.model_price.input_token_cost_per_million*response.total_tokens/1000000
            span.set_outputs({"status":"Embedding Generated", "query_cost":query_cost})
            return tuple(response.embeddings[0])