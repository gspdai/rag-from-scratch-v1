from enum import Enum

from .chunker import PageChunker
from .loader import load_document
from .vectordb import VectorDB
from .vectorizer import Vectorizer


class Status(Enum):
    ADDED = 1
    REPLACED = 2
    EXISTS = 3

class Insert():
    def __init__(self, chunker:PageChunker, vectorizer:Vectorizer, vectordb:VectorDB):
        self.chunker = chunker
        self.vectorizer = vectorizer
        self.vectordb = vectordb

    def add_doc(self, file:str) -> Status:
        doc = load_document(file)
        check_exist = self.vectordb.check_exists(doc.doc_name)
        if doc.doc_hash == check_exist:
            return Status.EXISTS
        elif not check_exist:
            chunk = self.chunker.chunk(doc)
            vector = self.vectorizer.doc(chunk)
            self.vectordb.add_doc(vector)
            return Status.ADDED
        else:
            chunk = self.chunker.chunk(doc)            
            vector = self.vectorizer.doc(chunk)
            self.vectordb.del_doc(doc.doc_name)
            self.vectordb.add_doc(vector)
            return Status.REPLACED
