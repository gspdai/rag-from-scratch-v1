from dataclasses import dataclass

from chromadb import PersistentClient

from .chunker import Chunk
from .vectorizer import Vector


@dataclass(frozen=True)
class TopMatch:
    chunk:Chunk
    similarity:float


class VectorDB():
    def __init__(self, location:str, db_name:str, similarity:str = "cosine"):
        self.client = PersistentClient(path=location)
        self.collection = self.client.get_or_create_collection(name=db_name,
                                                               metadata={"hnsw:space": similarity})

    def add_doc(self, doc_vecs:list[Vector]):
        id = [doc_vec.chunk.id for doc_vec in doc_vecs]
        metadata = [{"source":doc_vec.chunk.doc, 
                     "source_hash":doc_vec.chunk.doc_hash,
                     "citation":",".join(str(citation) for citation in doc_vec.chunk.citation),
                     "chunker":doc_vec.chunk.chunker} 
                     for doc_vec in doc_vecs]
        document = [doc_vec.chunk.text for doc_vec in doc_vecs]
        embedding = [list(doc_vec.embedding) for doc_vec in doc_vecs]
        self.collection.add(ids = id,
                            embeddings = embedding,
                            documents = document,
                            metadatas=metadata)

    def del_doc(self, doc:str):
        self.collection.delete(where={"source":doc})

    def check_exists(self, doc:str):
        vector_doc = self.collection.get(where={"source":doc},limit=1)
        if len(vector_doc["ids"]) == 0:
            return False
        else:
            return vector_doc["metadatas"][0]["source_hash"]

    def search(self, doc:str, query_embed:tuple[float,...] , top_k: int) -> list[TopMatch]:
        results = self.collection.query(query_embeddings=[list(query_embed)],
                                        n_results=top_k,
                                        where={"source":doc})
        matches = []
        for chunk_id, text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = Chunk(
                id=chunk_id,
                chunker=meta["chunker"],
                doc=meta["source"],
                doc_hash=meta["source_hash"],
                text=text,
                citation=tuple(int(page) for page in meta["citation"].split(",")),
            )
            matches.append(TopMatch(chunk=chunk, similarity=1 - distance))
        return matches

