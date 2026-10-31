from enum import Enum

from .chunker import PageChunker
from .loader import load_document
import logging
import time
from .vectordb import VectorDB
from .vectorizer import Vectorizer


logger = logging.getLogger(__name__)


class Status(Enum):
    ADDED = 1
    REPLACED = 2
    EXISTS = 3
    FAILED = 4

class Insert():
    def __init__(self, chunker:PageChunker, vectorizer:Vectorizer, vectordb:VectorDB):
        self.chunker = chunker
        self.vectorizer = vectorizer
        self.vectordb = vectordb

    def add_doc(self, file:str) -> Status:
        start_time = time.perf_counter()
        try:
            doc = load_document(file)
            logger.debug(f"Event=load_document Doc={doc.doc_name} page_count={len(doc.pages)}")
            check_exist = self.vectordb.check_exists(doc.doc_name)
            if doc.doc_hash == check_exist:
                logger.info(f"Event=ingest Doc={doc.doc_name} status={Status.EXISTS.name}")
                return Status.EXISTS
            elif not check_exist:
                chunk = self.chunker.chunk(doc)
                vector = self.vectorizer.doc(chunk)
                self.vectordb.add_doc(vector)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                logger.info(f"Event=ingest Doc={doc.doc_name} status={Status.ADDED.name} time_taken={total_time}")
                return Status.ADDED
            else:
                chunk = self.chunker.chunk(doc)            
                vector = self.vectorizer.doc(chunk)
                self.vectordb.del_doc(doc.doc_name)
                self.vectordb.add_doc(vector)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                logger.info(f"Event=ingest Doc={doc.doc_name} status={Status.REPLACED.name} time_taken={total_time}")
                return Status.REPLACED
        except Exception as error:
            logger.exception(f"event=ingest file={file} status={Status.FAILED.name} error={type(error).__name__}")
            return Status.FAILED