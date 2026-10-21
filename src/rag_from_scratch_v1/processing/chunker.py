from dataclasses import dataclass

from rag_from_scratch_v1.processing import Doc


@dataclass(frozen=True)
class Chunk:
    id: str
    chunker:str
    doc:str
    doc_hash:str
    text: str
    citation:tuple[int,...]


class PageChunker():
    def __init__(self):
        self.chunker = "page"

    def chunk(self,doc:Doc) -> list[Chunk]:
        chunks = []
        for page in doc.pages:
            if not page.text:
                continue
            chunk_id = doc.doc_name+":"+self.chunker+":"+str(page.number)
            chunks.append(Chunk(id=chunk_id, 
                                chunker=self.chunker, 
                                doc=doc.doc_name,
                                doc_hash=doc.doc_hash, 
                                text=page.text, 
                                citation=(page.number,)))
        return chunks