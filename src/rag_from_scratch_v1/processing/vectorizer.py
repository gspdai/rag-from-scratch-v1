from dataclasses import dataclass

import voyageai

from .chunker import Chunk


@dataclass(frozen=True)
class Vector:
    chunk: Chunk
    embedding:tuple[float,...]

class Vectorizer():
    def __init__(self, client: voyageai.Client, model: str):
        self.client = client
        self.model = model


    def doc(self, chunks:list[Chunk]) -> list[Vector]:
        chunk_text = [chunk.text for chunk in chunks]
        response = self.client.embed(texts=chunk_text, model=self.model, input_type="document")
        vector = [Vector(chunk=chunk, embedding=tuple(embedding)) for chunk, embedding in zip(chunks,response.embeddings)]
        return vector


    def query(self, text:str) -> tuple[float,...]:
        response = self.client.embed(texts= [text], model=self.model, input_type="query")
        return tuple(response.embeddings[0])